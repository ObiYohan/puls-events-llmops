"""Étape DVC « promote » : trace l'évaluation dans MLflow et décide de la mise en production.

Aucun appel aux services payants : cette étape lit les sorties de « evaluate »
et ne parle qu'au serveur MLflow (DagsHub, variables ``MLFLOW_TRACKING_*``).

1. Un run MLflow par évaluation : paramètres RAG, hash du prompt, scores,
   tokens, coût estimé ($), commit Git et empreinte DVC de l'index.
2. Décision (``decide``) : la version candidate devient « champion » si
   l'évaluation est valide, si tous les seuils sont atteints et si aucune
   métrique ne recule de plus de ``promotion.tolerance`` face au champion
   (comparaison faite seulement à juge et jeu de référence identiques).
3. Si elle est acceptée : l'index est joint au run, enregistré comme nouvelle
   version du modèle ``mlflow.registered_model``, et reçoit l'alias champion.
   L'API télécharge ensuite ``models:/<modèle>@champion``.

Le rapport ``paths.promotion_report`` résume la décision (lu par la CI).

Usage : ``uv run python -m src.pipeline.promote`` (ou ``uv run dvc repro promote``).
"""

import hashlib
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import yaml  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

from src.config import PROJECT_ROOT, Params, get_params, get_secret  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.pipeline.evaluate import METRICS  # noqa: E402
from src.pipeline.utils import read_json, write_json  # noqa: E402

logger = logging.getLogger(__name__)

TRACKING_URI_VAR = "MLFLOW_TRACKING_URI"
# Paramètres qui doivent être identiques pour comparer deux évaluations.
COMPARABILITY_KEYS = ("judge_model", "judge_reasoning_effort", "reference_set_version")


@dataclass
class Decision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    compared_to_champion: bool = False


# --- Décision (fonction pure) --------------------------------------------------

def decide(candidate: dict[str, Any], champion: dict[str, Any] | None, params: Params) -> Decision:
    """Applique les règles de mise en production du CLAUDE.md (B4)."""
    if not candidate.get("valid"):
        return Decision(False, [f"évaluation invalide ({candidate.get('n_metric_errors')} métrique(s) en erreur)"])
    # Seuils relus dans params.yaml (et non le résultat figé au moment de l'évaluation).
    thresholds = params.evaluation.thresholds
    missed = [
        name for name in METRICS
        if candidate.get(name) is None or candidate[name] < getattr(thresholds, name)
    ]
    if missed:
        details = ", ".join(f"{name} {candidate.get(name)} < {getattr(thresholds, name)}" for name in missed)
        return Decision(False, [f"seuil(s) non atteint(s) : {details}"])
    if champion is None:
        return Decision(True, ["aucun champion : première version promue"])

    differing = [key for key in COMPARABILITY_KEYS if str(candidate.get(key)) != str(champion.get(key))]
    if differing:
        return Decision(True, [
            f"champion évalué avec d'autres {', '.join(differing)} : comparaison impossible, seuils seuls appliqués"
        ])

    tolerance = params.promotion.tolerance
    regressions = [
        f"{name} {candidate[name]} < champion {champion[name]} - {tolerance}"
        for name in METRICS
        if candidate[name] < float(champion[name]) - tolerance
    ]
    if regressions:
        return Decision(False, ["recul par rapport au champion : " + " ; ".join(regressions)], True)
    return Decision(True, [f"au moins aussi bon que le champion (tolérance {tolerance})"], True)


# --- Paramètres, coût, traçabilité ---------------------------------------------

def prompt_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run_params(params: Params, evaluation: dict[str, Any]) -> dict[str, Any]:
    """Configuration RAG qui a produit la version évaluée."""
    c, r, g, e = params.chunking, params.retrieval, params.generation, params.evaluation
    return {
        "chunking.strip_html": c.strip_html,
        "chunking.chunk_size": c.chunk_size,
        "chunking.chunk_overlap": c.chunk_overlap,
        "chunking.split_threshold": c.split_threshold,
        "embedding.model": params.embedding.model,
        "retrieval.k": r.k,
        "retrieval.candidate_factor": r.candidate_factor,
        "generation.model": g.model,
        "generation.temperature": g.temperature,
        "generation.max_tokens": g.max_tokens,
        "prompt_hash": prompt_hash(params.paths.system_prompt),
        "judge_model": e.judge_model,
        "judge_reasoning_effort": e.judge_reasoning_effort,
        "reference_set_version": evaluation["reference_set_version"],
        "n_cases": evaluation["n_cases"],
    }


def cost_usd(tokens: dict[str, dict[str, int]], params: Params) -> dict[str, float]:
    """Coût estimé de l'évaluation à partir des tokens comptés et de ``pricing``."""
    def price(model: str, n_in: int, n_out: int = 0) -> float:
        p = params.pricing[model]
        return (n_in * p.input + n_out * p.output) / 1_000_000

    gen, judge = tokens.get("generation", {}), tokens.get("judge", {})
    generation = price(params.generation.model, gen.get("prompt_tokens", 0), gen.get("completion_tokens", 0))
    embedding = price(params.embedding.model, gen.get("embedding_tokens", 0))
    judging = price(params.evaluation.judge_model, judge.get("prompt_tokens", 0), judge.get("completion_tokens", 0))
    return {
        "cost_usd_generation": round(generation + embedding, 6),
        "cost_usd_judge": round(judging, 6),
        "cost_usd_total": round(generation + embedding + judging, 6),
    }


def run_metrics(evaluation: dict[str, Any], params: Params) -> dict[str, float]:
    tokens = evaluation.get("tokens", {})
    metrics = {name: evaluation[name] for name in METRICS if evaluation.get(name) is not None}
    metrics |= {"error_rate": evaluation["error_rate"], "n_metric_errors": evaluation["n_metric_errors"]}
    for group, counts in tokens.items():
        metrics |= {f"tokens_{group}_{name}": value for name, value in counts.items()}
    return metrics | cost_usd(tokens, params)


def git_info() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    try:
        return {"git_commit": git("rev-parse", "HEAD").strip(), "git_dirty": str(bool(git("status", "--porcelain").strip()))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "inconnu", "git_dirty": "inconnu"}


def index_md5(lock_path: Path = PROJECT_ROOT / "dvc.lock") -> str:
    """Empreinte DVC de l'index (permet de le restaurer avec ``dvc checkout``)."""
    try:
        lock = yaml.safe_load(Path(lock_path).read_text(encoding="utf-8"))
        return str(lock["stages"]["build_index"]["outs"][0]["md5"])
    except (OSError, KeyError, IndexError, TypeError):
        return "inconnu"


# --- Registre MLflow -----------------------------------------------------------

def champion_summary(client: MlflowClient, params: Params) -> dict[str, Any] | None:
    """Scores et paramètres de comparaison du champion actuel, ou ``None``."""
    try:
        version = client.get_model_version_by_alias(params.mlflow.registered_model, params.mlflow.champion_alias)
    except MlflowException:
        return None  # modèle ou alias absents
    run = client.get_run(version.run_id)
    return {**run.data.params, **run.data.metrics, "version": version.version, "run_id": version.run_id}


def register_champion(client: MlflowClient, params: Params, run: Any) -> str:
    name = params.mlflow.registered_model
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name, description="Index FAISS + chunks du RAG Puls-Events")
    version = client.create_model_version(name, source=f"{run.info.artifact_uri}/index", run_id=run.info.run_id)
    client.set_registered_model_alias(name, params.mlflow.champion_alias, version.version)
    return str(version.version)


# --- Étape ---------------------------------------------------------------------

def _require_tracking_uri() -> None:
    if get_secret(TRACKING_URI_VAR) is None:  # charge aussi .env (identifiants MLflow)
        raise RuntimeError(f"{TRACKING_URI_VAR} absente : impossible de tracer l'évaluation")
    mlflow.set_tracking_uri(os.environ[TRACKING_URI_VAR])


def run() -> dict[str, Any]:
    params = get_params()
    _require_tracking_uri()
    evaluation = read_json(params.paths.eval_metrics)
    client = MlflowClient()
    mlflow.set_experiment(params.mlflow.experiment)

    champion = champion_summary(client, params)
    run_parameters = run_params(params, evaluation)
    decision = decide({**evaluation, **run_parameters}, champion, params)

    with mlflow.start_run(run_name=f"eval-{evaluation['evaluated_on']}") as active:
        mlflow.log_params(run_parameters)
        mlflow.log_metrics(run_metrics(evaluation, params))
        mlflow.set_tags({
            **git_info(),
            "index_md5": index_md5(),
            "evaluated_on": evaluation["evaluated_on"],
            "decision": "promoted" if decision.promote else "rejected",
            "decision_reasons": " | ".join(decision.reasons),
        })
        mlflow.log_artifact(str(params.paths.eval_metrics), artifact_path="evaluation")
        mlflow.log_artifact(str(params.paths.eval_cases), artifact_path="evaluation")
        version = None
        if decision.promote:
            mlflow.log_artifacts(str(params.paths.index_dir), artifact_path="index")
            version = register_champion(client, params, active)

    report = {
        **asdict(decision),
        "run_id": active.info.run_id,
        "registered_model": params.mlflow.registered_model,
        "new_version": version,
        "previous_champion": None if champion is None else str(champion["version"]),
    }
    write_json(params.paths.promotion_report, report)
    logger.info("promote_done", extra=report)
    return report


if __name__ == "__main__":
    setup_logging()
    run()
