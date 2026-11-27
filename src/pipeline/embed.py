"""Étape DVC « embed » : embeddings incrémentaux (Azure OpenAI).

Le dossier de sortie (``paths.embeddings``) sert aussi de cache, conservé entre
deux exécutions grâce à ``persist: true`` dans dvc.yaml :

- ``keys.json``   : hash SHA-256 des textes enrichis, triés ;
- ``vectors.npy`` : vecteurs float32, dans le même ordre que ``keys.json`` ;
- ``meta.json``   : modèle et dimension.

Seuls les textes absents du cache sont envoyés à l'API ; changer de modèle
invalide tout le cache. Le cache est élagué aux chunks courants, et il est
sauvegardé même si l'exécution échoue en cours de route : les lots déjà payés
ne sont pas réencodés au prochain lancement.

Usage : ``uv run python -m src.pipeline.embed`` (ou ``uv run dvc repro embed``).
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import EmbeddingParams, get_params
from src.llm import is_retryable, make_client, status_code
from src.logging_utils import setup_logging
from src.pipeline.utils import count_tokens, read_json, write_json

logger = logging.getLogger(__name__)

KEYS_FILE = "keys.json"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


# --- Cache -------------------------------------------------------------------

def load_cache(directory: Path, model: str) -> dict[str, np.ndarray]:
    """Charge le cache ; renvoie un cache vide s'il est absent, incohérent ou d'un autre modèle."""
    meta_path = directory / META_FILE
    if not meta_path.exists():
        return {}
    meta = read_json(meta_path)
    if meta.get("model") != model:
        logger.info("embedding_cache_invalidated", extra={"cached_model": meta.get("model"), "model": model})
        return {}
    keys = read_json(directory / KEYS_FILE)
    vectors = np.load(directory / VECTORS_FILE, allow_pickle=False)
    if len(keys) != len(vectors):
        logger.warning("embedding_cache_corrupted", extra={"n_keys": len(keys), "n_vectors": len(vectors)})
        return {}
    return dict(zip(keys, vectors))


def save_cache(directory: Path, model: str, vectors: dict[str, np.ndarray]) -> None:
    """Écrit le cache de façon déterministe (clés triées)."""
    directory.mkdir(parents=True, exist_ok=True)
    keys = sorted(vectors)
    matrix = np.stack([vectors[k] for k in keys]).astype(np.float32) if keys else np.empty((0, 0), np.float32)
    write_json(directory / KEYS_FILE, keys)
    np.save(directory / VECTORS_FILE, matrix, allow_pickle=False)
    write_json(directory / META_FILE, {"model": model, "dim": int(matrix.shape[1]), "n_vectors": len(keys)})


# --- Appels API --------------------------------------------------------------

def make_batches(items: list[tuple[str, int]], max_tokens: int) -> list[list[str]]:
    """Regroupe des (clé, nb de tokens) en lots ne dépassant pas ``max_tokens``.

    Un texte plus long que la limite forme un lot à lui seul.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for key, tokens in items:
        if current and current_tokens + tokens > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(key)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def embed_batch(client: Any, texts: list[str], params: EmbeddingParams) -> tuple[list[list[float]], int]:
    """Encode un lot avec retry exponentiel ; renvoie (vecteurs, tokens facturés)."""
    for attempt in range(params.max_retries + 1):
        try:
            response = client.embeddings.create(model=params.model, input=texts)
            break
        except Exception as exc:
            if not is_retryable(exc) or attempt == params.max_retries:
                raise
            delay = params.retry_base_seconds * 2**attempt
            logger.warning(
                "embed_retry",
                extra={"attempt": attempt + 1, "status_code": status_code(exc), "delay_seconds": delay},
            )
            time.sleep(delay)

    data = sorted(response.data, key=lambda item: item.index)
    if len(data) != len(texts):
        raise ValueError(f"{len(data)} embeddings reçus pour {len(texts)} textes")
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    return [item.embedding for item in data], tokens


# --- Étape -------------------------------------------------------------------

def run(client: Any | None = None) -> dict[str, Any]:
    params = get_params()
    embedding = params.embedding
    tokenizer = params.chunking.tokenizer
    chunks = read_json(params.paths.chunks_dataset)

    # Deux chunks au texte enrichi identique partagent le même vecteur.
    texts = {c["text_hash"]: c["text_for_embedding"] for c in chunks}
    cache = load_cache(params.paths.embeddings, embedding.model)
    vectors = {key: cache[key] for key in texts if key in cache}
    missing = sorted(key for key in texts if key not in cache)

    stats = {
        "model": embedding.model,
        "n_chunks": len(chunks),
        "n_unique_texts": len(texts),
        "n_reused": len(vectors),
        "n_embedded": 0,
        "n_api_calls": 0,
        "prompt_tokens": 0,
    }
    logger.info("embed_start", extra={k: stats[k] for k in ("n_unique_texts", "n_reused")} | {"n_missing": len(missing)})

    try:
        if missing:
            client = client or make_client(max_retries=0)  # backoff géré ici
            batches = make_batches(
                [(key, count_tokens(texts[key], tokenizer)) for key in missing], embedding.max_tokens_per_batch
            )
            for i, batch in enumerate(batches):
                if i:
                    time.sleep(embedding.pause_seconds)
                batch_vectors, tokens = embed_batch(client, [texts[key] for key in batch], embedding)
                for key, vector in zip(batch, batch_vectors):
                    vectors[key] = np.asarray(vector, dtype=np.float32)
                stats["n_embedded"] += len(batch)
                stats["n_api_calls"] += 1
                stats["prompt_tokens"] += tokens
                logger.info(
                    "embed_batch_done",
                    extra={"batch": i + 1, "n_batches": len(batches), "n_texts": len(batch), "prompt_tokens": tokens},
                )
    finally:
        # Sauvegarde même en cas d'échec : le travail déjà payé est conservé.
        save_cache(params.paths.embeddings, embedding.model, vectors)

    write_json(params.paths.embed_metrics, stats)
    logger.info("embed_done", extra=stats)
    return stats


if __name__ == "__main__":
    setup_logging()
    run()
