"""Résumé Markdown d'une exécution du pipeline (page d'exécution GitHub Actions).

Lit les métriques produites par DVC (index, évaluation, promotion) et écrit un
tableau dans ``$GITHUB_STEP_SUMMARY`` (ou sur la sortie standard en local).
Si la nouvelle version n'est pas promue, émet un avertissement GitHub
(``::warning::``) : un refus n'est pas un échec du pipeline, mais il doit se voir.

Usage : ``uv run python -m src.pipeline.summary``.
"""

import os
import sys
from pathlib import Path
from typing import Any

from src.config import get_params
from src.pipeline.evaluate import METRICS
from src.pipeline.utils import read_json


def _read(path: Path) -> dict[str, Any] | None:
    return read_json(path) if Path(path).exists() else None


def render(index: dict | None, evaluation: dict | None, promotion: dict | None, thresholds: Any) -> str:
    lines = ["## Pipeline RAG Puls-Events", ""]
    if index:
        lines += [
            f"**Index** : {index['n_vectors']} chunks, {index['n_events']} événements, "
            f"modèle `{index['embedding_model']}` ({index['dim']} dimensions).",
            "",
        ]
    if evaluation:
        lines += [
            f"**Évaluation** du {evaluation['evaluated_on']} — juge `{evaluation['judge_model']}` "
            f"({evaluation['judge_reasoning_effort']}), {evaluation['n_cases']} questions, "
            f"{'valide' if evaluation['valid'] else '**invalide**'} "
            f"({evaluation['n_metric_errors']} métrique(s) en erreur).",
            "",
            "| Métrique | Score | Seuil |",
            "|---|---|---|",
        ]
        for name in METRICS:
            score, threshold = evaluation.get(name), getattr(thresholds, name)
            mark = "✅" if score is not None and score >= threshold else "❌"
            lines.append(f"| {name} | {score} {mark} | {threshold} |")
        lines.append("")
    if promotion:
        verdict = (
            f"✅ **promue** : version {promotion['new_version']} de `{promotion['registered_model']}` (alias champion)"
            if promotion["promote"]
            else f"⛔ **refusée** : le champion reste la version {promotion['previous_champion']}"
        )
        lines += [f"**Mise en production** : {verdict}.", ""]
        lines += [f"- {reason}" for reason in promotion["reasons"]]
    return "\n".join(lines) + "\n"


def run() -> str:
    params = get_params()
    paths = params.paths
    promotion = _read(paths.promotion_report)
    text = render(_read(paths.index_metrics), _read(paths.eval_metrics), promotion, params.evaluation.thresholds)

    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(text)
        if promotion and not promotion["promote"]:
            print(f"::warning::Version non promue : {' | '.join(promotion['reasons'])}")
    else:
        sys.stdout.write(text)
    return text


if __name__ == "__main__":
    run()
