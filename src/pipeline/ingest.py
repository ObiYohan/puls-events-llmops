"""Étape DVC « ingest » : collecte des événements OpenAgenda via l'API OpenDataSoft.

- Ne télécharge que les événements de la région configurée qui ne sont pas
  terminés à la date de collecte (filtre appliqué côté serveur).
- Ne conserve que les champs de la liste blanche ``ingest.fields`` : aucune
  donnée personnelle des contributeurs n'entre dans le pipeline.
- Écrit un fichier trié par ``uid`` avec des clés ordonnées : deux collectes
  le même jour sur les mêmes données donnent un fichier identique, donc le même
  hash DVC, et les étapes suivantes ne sont pas relancées pour rien.

Usage : ``uv run python -m src.pipeline.ingest`` (ou ``uv run dvc repro ingest``).
"""

import logging
import time
from datetime import UTC, date, datetime
from typing import Any

import httpx

from src.config import IngestParams, get_params
from src.logging_utils import setup_logging
from src.pipeline.utils import write_json

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def build_query(params: IngestParams, snapshot_date: date) -> dict[str, str]:
    """Paramètres ODSQL de l'export : région + événements non terminés + champs."""
    region = params.region.replace('"', '\\"')
    where = f"location_region=\"{region}\" and lastdate_end >= date'{snapshot_date.isoformat()}'"
    return {"where": where, "select": ",".join(params.fields)}


def fetch_events(
    params: IngestParams, snapshot_date: date, client: httpx.Client | None = None
) -> list[dict[str, Any]]:
    """Télécharge l'export JSON, avec retry exponentiel sur 429, 5xx et erreurs réseau."""
    url = f"{params.base_url.rstrip('/')}/exports/json"
    query = build_query(params, snapshot_date)
    owns_client = client is None
    client = client or httpx.Client(timeout=params.timeout_seconds)
    try:
        for attempt in range(params.max_retries + 1):
            try:
                response = client.get(url, params=query)
                if response.status_code not in RETRYABLE_STATUS:
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, list):
                        raise ValueError("réponse inattendue de l'API : une liste JSON était attendue")
                    return data
                reason = f"HTTP {response.status_code}"
            except httpx.TransportError as exc:
                reason = type(exc).__name__

            if attempt == params.max_retries:
                raise RuntimeError(f"échec de la collecte après {attempt + 1} tentatives ({reason})")
            delay = params.retry_base_seconds * 2**attempt
            logger.warning(
                "ingest_retry", extra={"attempt": attempt + 1, "reason": reason, "delay_seconds": delay}
            )
            time.sleep(delay)
    finally:
        if owns_client:
            client.close()
    raise AssertionError("unreachable")  # pragma: no cover


def _has_description(event: dict[str, Any]) -> bool:
    text = event.get("longdescription_fr")
    return isinstance(text, str) and bool(text.strip())


def clean_events(
    raw_events: list[dict[str, Any]], fields: list[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Applique la liste blanche, écarte les événements sans description, déduplique et trie."""
    allowed = set(fields)
    stats = {"n_received": len(raw_events), "n_without_description": 0, "n_duplicates": 0}

    by_uid: dict[str, dict[str, Any]] = {}
    for raw in raw_events:
        if not _has_description(raw):
            stats["n_without_description"] += 1
            continue
        event = {k: v for k, v in raw.items() if k in allowed}
        uid = str(event["uid"])
        previous = by_uid.get(uid)
        if previous is not None:
            stats["n_duplicates"] += 1
            # On garde la version la plus récemment mise à jour.
            if str(event.get("updatedat") or "") <= str(previous.get("updatedat") or ""):
                continue
        by_uid[uid] = event

    events = [by_uid[uid] for uid in sorted(by_uid)]
    stats["n_kept"] = len(events)
    return events, stats


def run(snapshot_date: date | None = None) -> dict[str, Any]:
    """Exécute l'étape complète et renvoie les métriques écrites."""
    params = get_params()
    snapshot_date = snapshot_date or datetime.now(UTC).date()
    logger.info("ingest_start", extra={"region": params.ingest.region, "snapshot_date": snapshot_date.isoformat()})

    raw_events = fetch_events(params.ingest, snapshot_date)
    events, stats = clean_events(raw_events, params.ingest.fields)
    if not events:
        # Échec volontaire : DVC n'enregistre pas de nouvelle version et la
        # dernière version valide reste récupérable (dvc checkout).
        raise RuntimeError("aucun événement exploitable : arrêt du pipeline")

    write_json(params.paths.raw_events, events)
    metrics = {"snapshot_date": snapshot_date.isoformat(), **stats}
    write_json(params.paths.ingest_metrics, metrics)
    logger.info("ingest_done", extra=metrics)
    return metrics


if __name__ == "__main__":
    setup_logging()
    run()
