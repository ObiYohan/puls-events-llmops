"""Étape DVC « evaluate » : évaluation DeepEval du RAG sur le jeu de référence (B4).

Pour chaque question du jeu de référence (``paths.reference_set``) :

1. le chatbot répond, avec l'index courant (``paths.index_dir``) et le modèle de
   génération de ``params.yaml`` ;
2. le juge (``evaluation.judge_model``, distinct du générateur) note la réponse
   avec quatre métriques DeepEval : Faithfulness, AnswerRelevancy,
   ContextualPrecision et ContextualRecall.

Sorties : ``paths.eval_metrics`` (moyennes, seuils, tokens, validité) et
``paths.eval_cases`` (détail par question). L'évaluation est **invalide** si la
part de métriques en erreur dépasse ``evaluation.max_error_rate`` : une
évaluation incomplète ne doit jamais permettre une mise en production.

Usage : ``uv run python -m src.pipeline.evaluate`` (ou ``uv run dvc repro evaluate``).
"""

import os

# Avant l'import de deepeval : pas de télémétrie ni de connexion à Confident AI.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import asyncio  # noqa: E402
import logging  # noqa: E402
import statistics  # noqa: E402
import threading  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import openai  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from src.config import EvaluationParams, get_params  # noqa: E402
from src.llm import make_client, with_retry  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.pipeline.chunk import FRENCH_MONTHS  # noqa: E402
from src.pipeline.utils import read_json, write_json  # noqa: E402

logger = logging.getLogger(__name__)

METRICS = {
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
}

NEXT_MONTH = "{mois_prochain}"
NO_CONTEXT = "Aucun événement trouvé."
# Séparateur placé par EventChatbot.format_context devant le premier événement.
CONTEXT_SEPARATOR = "\n\n" + "=" * 80


# --- Jeu de référence ----------------------------------------------------------

def next_month_name(today: date) -> str:
    return FRENCH_MONTHS[today.month % 12]


def load_reference_set(path: Path, today: date) -> tuple[int, list[dict[str, str]]]:
    """Charge le jeu de référence et remplace les dates relatives ; renvoie (version, cas)."""
    data = read_json(path)
    cases = data["cases"]
    ids = [c["id"] for c in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("jeu de référence vide ou identifiants en double")
    month = next_month_name(today)
    resolved = [
        {
            **case,
            "question": case["question"].replace(NEXT_MONTH, month),
            "expected_output": case["expected_output"].replace(NEXT_MONTH, month),
        }
        for case in cases
    ]
    return int(data["version"]), resolved


# --- Juge ----------------------------------------------------------------------

class OpenAIJudge(DeepEvalBaseLLM):
    """Juge DeepEval sur un client OpenAI (Azure v1), en sorties structurées.

    DeepEval passe le schéma Pydantic attendu par chaque métrique : la réponse
    est contrainte par ce schéma, ce qui évite le JSON invalide. Si le service
    refuse le schéma, repli sur le mode JSON simple (DeepEval parse le texte).
    Les tokens consommés sont comptés (appels concurrents : verrou).
    """

    def __init__(self, client: Any, params: EvaluationParams):
        self.client = client
        self.params = params
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": 0}
        self._lock = threading.Lock()
        super().__init__(params.judge_model)

    def load_model(self) -> Any:
        return self.client

    def get_model_name(self) -> str:
        return self.params.judge_model

    def _record(self, usage: Any) -> None:
        details = getattr(usage, "completion_tokens_details", None)
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            self.usage["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)

    def _call(self, fn, name: str):
        return with_retry(
            fn, max_retries=self.params.max_retries, base_seconds=self.params.retry_base_seconds, name=name
        )

    def generate(self, prompt: str, schema: Any = None) -> Any:
        kwargs = {
            "model": self.params.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": self.params.judge_reasoning_effort,
        }
        if schema is not None:
            try:
                response = self._call(
                    lambda: self.client.chat.completions.parse(response_format=schema, **kwargs), "judge_parse"
                )
                self._record(response.usage)
                message = response.choices[0].message
                return message.parsed if message.parsed is not None else message.content
            except openai.BadRequestError:
                logger.warning("judge_schema_rejected", extra={"schema": getattr(schema, "__name__", str(schema))})
                kwargs["response_format"] = {"type": "json_object"}
        response = self._call(lambda: self.client.chat.completions.create(**kwargs), "judge_create")
        self._record(response.usage)
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)


# --- Réponses du RAG -----------------------------------------------------------

def retrieval_context(chatbot: Any, chunks: Any) -> list[str]:
    """Un bloc de contexte par événement, dans l'ordre du classement (ContextualPrecision en dépend)."""
    if chunks is None or len(chunks) == 0:
        return [NO_CONTEXT]
    blocks = [chatbot.format_context(group) for _, group in chunks.groupby("uid", sort=False)]
    return [block.removeprefix(CONTEXT_SEPARATOR) for block in blocks]


def answer_case(chatbot: Any, case: dict[str, str], params: EvaluationParams) -> dict[str, Any]:
    question = case["question"]
    chunks, _ = chatbot.search_relevant_events(question)
    context = chatbot.format_context(chunks)
    answer = with_retry(
        lambda: chatbot.generate_response(question, context),
        max_retries=params.max_retries, base_seconds=params.retry_base_seconds, name="generation",
    )
    n_events = 0 if chunks is None or len(chunks) == 0 else int(chunks["uid"].nunique())
    return {**case, "answer": answer, "retrieval_context": retrieval_context(chatbot, chunks), "n_events": n_events}


# --- Notation ------------------------------------------------------------------

def judge_case(case: dict[str, Any], judge: DeepEvalBaseLLM, params: EvaluationParams) -> dict[str, Any]:
    """Note un cas avec les quatre métriques ; une métrique en erreur est notée ``None``."""
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=case["answer"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"],
    )
    scores, reasons, errors = {}, {}, {}
    for name, metric_cls in METRICS.items():
        metric = metric_cls(
            model=judge, threshold=getattr(params.thresholds, name), async_mode=False, include_reason=True
        )
        try:
            metric.measure(test_case)
            scores[name] = float(metric.score)
            reasons[name] = metric.reason
        except Exception as exc:  # une métrique en échec ne doit pas arrêter les autres
            scores[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("metric_failed", extra={"case": case["id"], "metric": name, "error": type(exc).__name__})
    return {**case, "scores": scores, "reasons": reasons, "errors": errors}


def aggregate(cases: list[dict[str, Any]], params: EvaluationParams) -> dict[str, Any]:
    """Moyennes par métrique, taux d'erreur, validité et respect des seuils."""
    n_scores = len(cases) * len(METRICS)
    n_errors = sum(1 for c in cases for v in c["scores"].values() if v is None)
    error_rate = n_errors / n_scores if n_scores else 1.0
    valid = bool(cases) and error_rate <= params.max_error_rate

    means: dict[str, float | None] = {}
    for name in METRICS:
        values = [c["scores"][name] for c in cases if c["scores"][name] is not None]
        means[name] = round(statistics.fmean(values), 4) if values else None
    thresholds_met = {
        name: means[name] is not None and means[name] >= getattr(params.thresholds, name) for name in METRICS
    }
    return {
        **means,
        "n_cases": len(cases),
        "n_metric_errors": n_errors,
        "error_rate": round(error_rate, 4),
        "valid": valid,
        "thresholds_met": thresholds_met,
        "passed": valid and all(thresholds_met.values()),
    }


# --- Étape ---------------------------------------------------------------------

def _make_chatbot():
    from src.chatbot import EventChatbot  # import tardif : charge l'index

    return EventChatbot()


def run(chatbot: Any = None, judge: DeepEvalBaseLLM | None = None, today: date | None = None) -> dict[str, Any]:
    params = get_params()
    evaluation = params.evaluation
    today = today or date.today()
    version, cases = load_reference_set(params.paths.reference_set, today)
    chatbot = chatbot or _make_chatbot()
    judge = judge or OpenAIJudge(make_client(max_retries=0), evaluation)
    logger.info("evaluate_start", extra={"n_cases": len(cases), "judge_model": evaluation.judge_model})

    # Génération séquentielle (compteurs de tokens du chatbot), notation en parallèle.
    answered = [answer_case(chatbot, case, evaluation) for case in cases]
    with ThreadPoolExecutor(max_workers=evaluation.max_workers) as pool:
        judged = list(pool.map(lambda case: judge_case(case, judge, evaluation), answered))

    metrics = {
        **aggregate(judged, evaluation),
        "reference_set_version": version,
        "evaluated_on": today.isoformat(),
        "generation_model": params.generation.model,
        "judge_model": evaluation.judge_model,
        "judge_reasoning_effort": evaluation.judge_reasoning_effort,
        "tokens": {
            "generation": dict(getattr(chatbot, "token_usage", {})),
            "judge": dict(getattr(judge, "usage", {})),
        },
    }
    write_json(params.paths.eval_cases, judged)
    write_json(params.paths.eval_metrics, metrics)
    logger.info("evaluate_done", extra={k: metrics[k] for k in (*METRICS, "valid", "passed", "error_rate")})
    return metrics


if __name__ == "__main__":
    setup_logging()
    run()
