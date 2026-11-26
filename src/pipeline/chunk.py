"""Étape DVC « chunk » : découpage des descriptions en chunks prêts à encoder.

Reprend la logique de Projet_7 :
- une description courte (≤ ``chunking.split_threshold`` tokens) reste un seul chunk ;
- une description longue est découpée en chunks de ``chunking.chunk_size`` tokens ;
- chaque chunk porte les métadonnées de date et de lieu de son événement ;
- le texte encodé est « enrichi » (ville, lieu, mois) pour améliorer la recherche.

Chaque chunk reçoit ``text_hash`` (SHA-256 du texte enrichi), la clé du cache
d'embeddings : un chunk dont le texte enrichi ne change pas n'est jamais réencodé.

Usage : ``uv run python -m src.pipeline.chunk`` (ou ``uv run dvc repro chunk``).
"""

import logging
import statistics
from datetime import datetime
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkingParams, get_params
from src.logging_utils import setup_logging
from src.pipeline.utils import count_tokens, read_json, text_hash, write_json

logger = logging.getLogger(__name__)

# Métadonnées de l'événement recopiées dans chaque chunk (utilisées par les filtres du chatbot).
CHUNK_METADATA_FIELDS = (
    "firstdate_begin", "firstdate_end", "lastdate_begin", "lastdate_end",
    "location_name", "location_city", "location_postalcode", "location_phone", "location_website",
)

FRENCH_MONTHS = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def enriched_text(text: str, metadata: dict[str, Any]) -> str:
    """Texte envoyé au modèle d'embedding : description + ville + lieu + mois."""
    parts = [text]
    if _present(metadata.get("location_city")):
        parts.append(f"Ville: {metadata['location_city']}")
    if _present(metadata.get("location_name")):
        parts.append(f"Lieu: {metadata['location_name']}")
    begin = metadata.get("firstdate_begin")
    if _present(begin):
        try:
            # Mois dans le fuseau de l'événement (l'offset ISO est conservé).
            date = datetime.fromisoformat(str(begin))
            parts.append(f"Date: {FRENCH_MONTHS[date.month - 1]} {date.year}")
        except ValueError:
            logger.warning("invalid_date", extra={"field": "firstdate_begin"})
    return " | ".join(parts)


def make_splitter(params: ChunkingParams) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=params.chunk_size,
        chunk_overlap=params.chunk_overlap,
        length_function=lambda text: count_tokens(text, params.tokenizer),
        separators=params.separators,
    )


def chunk_event(
    event: dict[str, Any], params: ChunkingParams, splitter: RecursiveCharacterTextSplitter
) -> list[dict[str, Any]]:
    """Découpe un événement en un ou plusieurs chunks porteurs de ses métadonnées."""
    description = event["longdescription_fr"]
    if count_tokens(description, params.tokenizer) <= params.split_threshold:
        texts = [description]
    else:
        texts = splitter.split_text(description)

    metadata = {field: event.get(field) for field in CHUNK_METADATA_FIELDS}
    chunks = []
    for chunk_id, text in enumerate(texts):
        text_for_embedding = enriched_text(text, metadata)
        chunks.append({
            "uid": str(event["uid"]),
            "chunk_id": chunk_id,
            "text": text,
            "token_count": count_tokens(text, params.tokenizer),
            "text_for_embedding": text_for_embedding,
            "text_hash": text_hash(text_for_embedding),
            **metadata,
        })
    return chunks


def chunk_events(events: list[dict[str, Any]], params: ChunkingParams) -> list[dict[str, Any]]:
    splitter = make_splitter(params)
    chunks = [chunk for event in events for chunk in chunk_event(event, params, splitter)]
    chunks.sort(key=lambda c: (c["uid"], c["chunk_id"]))
    return chunks


def compute_metrics(events: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [c["token_count"] for c in chunks]
    return {
        "n_events": len(events),
        "n_chunks": len(chunks),
        "n_split_events": len({c["uid"] for c in chunks if c["chunk_id"] > 0}),
        "chunk_tokens_mean": round(statistics.fmean(tokens), 1) if tokens else 0.0,
        "chunk_tokens_max": max(tokens, default=0),
        "chunk_tokens_total": sum(tokens),
    }


def run() -> dict[str, Any]:
    params = get_params()
    events = read_json(params.paths.raw_events)
    chunks = chunk_events(events, params.chunking)
    metrics = compute_metrics(events, chunks)
    write_json(params.paths.chunks_dataset, chunks)
    write_json(params.paths.chunk_metrics, metrics)
    logger.info("chunk_done", extra=metrics)
    return metrics


if __name__ == "__main__":
    setup_logging()
    run()
