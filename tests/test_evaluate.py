import json
import re
from datetime import date
from types import SimpleNamespace

import httpx
import openai
import pandas as pd
import pytest
from pydantic import BaseModel

from src.config import PROJECT_ROOT, get_params
from src.pipeline import evaluate
from src.pipeline.utils import read_json, write_json


# --- Jeu de référence ----------------------------------------------------------

@pytest.mark.parametrize(("today", "expected"), [
    (date(2026, 11, 27), "décembre"),
    (date(2026, 12, 3), "janvier"),
    (date(2027, 1, 31), "février"),
])
def test_next_month_name(today, expected):
    assert evaluate.next_month_name(today) == expected


def test_reference_set_placeholders_are_resolved(tmp_path):
    path = tmp_path / "ref.json"
    write_json(path, {"version": 3, "cases": [
        {"id": "a", "question": "Que faire en {mois_prochain} ?", "expected_output": "Des événements en {mois_prochain}."},
    ]})
    version, cases = evaluate.load_reference_set(path, date(2026, 11, 27))
    assert version == 3
    assert cases[0]["question"] == "Que faire en décembre ?"
    assert cases[0]["expected_output"] == "Des événements en décembre."


def test_reference_set_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "ref.json"
    case = {"id": "a", "question": "q", "expected_output": "e"}
    write_json(path, {"version": 1, "cases": [case, case]})
    with pytest.raises(ValueError, match="double"):
        evaluate.load_reference_set(path, date(2026, 11, 27))


def test_project_reference_set_has_no_absolute_dates():
    """Garde-fou du CLAUDE.md : aucune année ni date absolue dans le jeu de référence."""
    raw = get_params().paths.reference_set.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert 20 <= len(data["cases"]) <= 30
    texts = " ".join(c["question"] + " " + c["expected_output"] for c in data["cases"])
    assert not re.search(r"\b(19|20)\d{2}\b", texts)
    assert not re.search(r"\b\d{1,2}/\d{1,2}\b", texts)
    for month in evaluate.FRENCH_MONTHS:
        assert not re.search(rf"\b{month}\b", texts, re.IGNORECASE), month


# --- Juge ----------------------------------------------------------------------

class Verdict(BaseModel):
    verdict: str


def completion(parsed=None, content="{}", prompt=100, completion_tokens=20, reasoning=5):
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    message = SimpleNamespace(parsed=parsed, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeCompletions:
    def __init__(self, parse_error=None):
        self.parse_calls, self.create_calls = [], []
        self.parse_error = parse_error

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if self.parse_error:
            raise self.parse_error
        return completion(parsed=kwargs["response_format"](verdict="yes"))

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return completion(content='{"verdict": "no"}')


def fake_client(**kwargs):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(**kwargs)))


@pytest.fixture
def eval_params():
    return get_params().evaluation


def test_judge_uses_structured_output_and_counts_tokens(eval_params):
    client = fake_client()
    judge = evaluate.OpenAIJudge(client, eval_params)

    result = judge.generate("note", schema=Verdict)

    assert result == Verdict(verdict="yes")
    call = client.chat.completions.parse_calls[0]
    assert call["model"] == eval_params.judge_model
    assert call["reasoning_effort"] == eval_params.judge_reasoning_effort
    assert judge.usage == {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 5, "calls": 1}


def test_judge_without_schema_returns_text(eval_params):
    client = fake_client()
    assert evaluate.OpenAIJudge(client, eval_params).generate("note") == '{"verdict": "no"}'
    assert client.chat.completions.parse_calls == []


def test_judge_falls_back_to_json_mode_when_schema_rejected(eval_params):
    request = httpx.Request("POST", "https://test")
    error = openai.BadRequestError("schema", response=httpx.Response(400, request=request), body=None)
    client = fake_client(parse_error=error)

    result = evaluate.OpenAIJudge(client, eval_params).generate("note", schema=Verdict)

    assert result == '{"verdict": "no"}'
    assert client.chat.completions.create_calls[0]["response_format"] == {"type": "json_object"}


def test_judge_name_is_judge_model(eval_params):
    assert evaluate.OpenAIJudge(fake_client(), eval_params).get_model_name() == eval_params.judge_model


# --- Contexte et agrégation ----------------------------------------------------

class FakeChatbot:
    def __init__(self, chunks):
        self.chunks = chunks
        self.token_usage = {"embedding_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def search_relevant_events(self, question):
        return self.chunks, None

    def format_context(self, df):
        if df is None or len(df) == 0:
            return "Aucun événement trouvé."
        return "\n\n" + "=" * 80 + "\n\n".join(f"=== ÉVÉNEMENT {uid} ===" for uid in df["uid"].unique())

    def generate_response(self, question, context):
        self.token_usage["prompt_tokens"] += 10
        self.token_usage["completion_tokens"] += 2
        return f"Réponse à : {question}"


def test_retrieval_context_keeps_rank_order_one_block_per_event():
    chunks = pd.DataFrame({"uid": ["z", "z", "a"], "chunk_id": [0, 1, 0], "text": ["t1", "t2", "t3"]})
    blocks = evaluate.retrieval_context(FakeChatbot(chunks), chunks)
    assert blocks == ["=== ÉVÉNEMENT z ===", "=== ÉVÉNEMENT a ==="]


def test_retrieval_context_when_nothing_found():
    assert evaluate.retrieval_context(FakeChatbot(pd.DataFrame()), pd.DataFrame()) == [evaluate.NO_CONTEXT]


def scored(**scores):
    return {"scores": {name: scores.get(name, 0.9) for name in evaluate.METRICS}}


def test_aggregate_passes_when_valid_and_above_thresholds(eval_params):
    result = evaluate.aggregate([scored(), scored()], eval_params)
    assert result["faithfulness"] == 0.9
    assert result["valid"] and result["passed"]
    assert result["error_rate"] == 0


def test_aggregate_fails_on_threshold(eval_params):
    low = eval_params.thresholds.faithfulness - 0.1
    result = evaluate.aggregate([scored(faithfulness=low)], eval_params)
    assert result["valid"] and not result["passed"]
    assert result["thresholds_met"]["faithfulness"] is False


def test_aggregate_is_invalid_when_a_metric_errored(eval_params):
    result = evaluate.aggregate([scored(), scored(contextual_recall=None)], eval_params)
    assert result["n_metric_errors"] == 1
    assert not result["valid"] and not result["passed"]
    assert result["contextual_recall"] == 0.9  # moyenne sur les scores disponibles


def test_aggregate_of_empty_run_is_invalid(eval_params):
    assert evaluate.aggregate([], eval_params)["valid"] is False


# --- Étape complète, métriques simulées --------------------------------------

class FakeMetric:
    def __init__(self, model, threshold, async_mode, include_reason):
        self.model = model

    def measure(self, test_case):
        if "boom" in test_case.input:
            raise RuntimeError("juge indisponible")
        self.score = 0.95
        self.reason = "ok"


def test_run_writes_metrics_and_cases(isolated_params, monkeypatch):
    params = isolated_params.model_copy(update={
        "paths": isolated_params.paths.model_copy(update={"reference_set": PROJECT_ROOT / "tests/eval/reference_set.json"}),
    })
    monkeypatch.setattr(evaluate, "get_params", lambda: params)
    monkeypatch.setattr(evaluate, "METRICS", {name: FakeMetric for name in evaluate.METRICS})
    chunks = pd.DataFrame({"uid": ["e1"], "chunk_id": [0], "text": ["Concert"]})
    judge = evaluate.OpenAIJudge(fake_client(), params.evaluation)

    metrics = evaluate.run(chatbot=FakeChatbot(chunks), judge=judge, today=date(2026, 11, 27))

    assert metrics["passed"] and metrics["valid"]
    assert metrics["n_cases"] == 24
    assert metrics["evaluated_on"] == "2026-11-27"
    assert metrics["judge_model"] == params.evaluation.judge_model
    assert metrics["tokens"]["generation"]["prompt_tokens"] == 240
    assert read_json(params.paths.eval_metrics) == metrics
    cases = read_json(params.paths.eval_cases)
    assert "décembre" in next(c for c in cases if c["id"] == "mois-marseille")["question"]
    assert all(c["retrieval_context"] == ["=== ÉVÉNEMENT e1 ==="] for c in cases)


def test_run_with_failing_metric_is_invalid(isolated_params, monkeypatch, tmp_path):
    ref = tmp_path / "ref.json"
    write_json(ref, {"version": 1, "cases": [
        {"id": "ok", "question": "Que faire ?", "expected_output": "Des événements."},
        {"id": "ko", "question": "boom", "expected_output": "Des événements."},
    ]})
    params = isolated_params.model_copy(update={"paths": isolated_params.paths.model_copy(update={"reference_set": ref})})
    monkeypatch.setattr(evaluate, "get_params", lambda: params)
    monkeypatch.setattr(evaluate, "METRICS", {name: FakeMetric for name in evaluate.METRICS})
    chunks = pd.DataFrame({"uid": ["e1"], "chunk_id": [0], "text": ["Concert"]})

    metrics = evaluate.run(chatbot=FakeChatbot(chunks), judge=evaluate.OpenAIJudge(fake_client(), params.evaluation))

    assert metrics["n_metric_errors"] == 4
    assert not metrics["valid"] and not metrics["passed"]
    failed = next(c for c in read_json(params.paths.eval_cases) if c["id"] == "ko")
    assert set(failed["errors"]) == set(evaluate.METRICS)
