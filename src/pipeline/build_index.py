"""Étape DVC « build_index » : assemble l'artefact servi par l'API.

Le dossier ``paths.index_dir`` est l'unité versionnée (DVC) et promue (MLflow) :

- ``faiss.index``   : vecteurs des chunks, la ligne i de l'index = le chunk i ;
- ``chunks.json``   : texte et métadonnées des chunks, dans le même ordre ;
- ``manifest.json`` : modèle d'embedding, dimension et volumes, pour vérifier au
  chargement que l'API encode les questions avec le même modèle que l'index.

Aucun appel réseau : les vecteurs viennent du cache de l'étape « embed ».

Usage : ``uv run python -m src.pipeline.build_index`` (ou ``uv run dvc repro build_index``).
"""

import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import get_params
from src.logging_utils import setup_logging
from src.pipeline.embed import load_cache
from src.pipeline.utils import read_json, write_json

logger = logging.getLogger(__name__)

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"

# Champs propres au pipeline, inutiles à l'API.
PIPELINE_ONLY_FIELDS = ("text_for_embedding", "text_hash")


def assemble(chunks: list[dict[str, Any]], cache: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Associe à chaque chunk son vecteur ; échoue si un vecteur manque."""
    missing = sorted({c["text_hash"] for c in chunks if c["text_hash"] not in cache})
    if missing:
        raise RuntimeError(f"{len(missing)} chunk(s) sans embedding : relancer l'étape « embed »")
    if not chunks:
        raise RuntimeError("aucun chunk à indexer")
    matrix = np.stack([cache[c["text_hash"]] for c in chunks]).astype(np.float32)
    records = [{k: v for k, v in c.items() if k not in PIPELINE_ONLY_FIELDS} for c in chunks]
    return matrix, records


def build_faiss_index(matrix: np.ndarray) -> faiss.Index:
    """Index exact en distance L2 (vecteurs normalisés : même classement que le cosinus)."""
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index


def write_artifact(directory: Path, index: faiss.Index, records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / INDEX_FILE))
    write_json(directory / CHUNKS_FILE, records)
    write_json(directory / MANIFEST_FILE, manifest)


def run() -> dict[str, Any]:
    params = get_params()
    model = params.embedding.model
    chunks = read_json(params.paths.chunks_dataset)
    cache = load_cache(params.paths.embeddings, model)

    matrix, records = assemble(chunks, cache)
    index = build_faiss_index(matrix)

    manifest = {
        "embedding_model": model,
        "dim": int(index.d),
        "n_vectors": int(index.ntotal),
        "n_events": len({r["uid"] for r in records}),
    }
    write_artifact(params.paths.index_dir, index, records, manifest)
    write_json(params.paths.index_metrics, manifest)
    logger.info("build_index_done", extra=manifest)
    return manifest


if __name__ == "__main__":
    setup_logging()
    run()
