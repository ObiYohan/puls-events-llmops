import faiss
import numpy as np
import pytest

from src.pipeline import build_index, embed
from src.pipeline.utils import read_json, text_hash, write_json


def chunk(uid, chunk_id, text):
    enriched = f"{text} | Ville: Nice"
    return {
        "uid": uid, "chunk_id": chunk_id, "text": text, "token_count": 3,
        "text_for_embedding": enriched, "text_hash": text_hash(enriched),
        "location_city": "Nice", "lastdate_end": "2099-12-05T23:00:00+01:00",
    }


@pytest.fixture
def params(isolated_params, monkeypatch):
    monkeypatch.setattr(build_index, "get_params", lambda: isolated_params)
    return isolated_params


def setup_pipeline_outputs(params, chunks, skip=()):
    write_json(params.paths.chunks_dataset, chunks)
    vectors = {
        c["text_hash"]: np.array([float(i), 1.0, 0.0], dtype=np.float32)
        for i, c in enumerate(chunks) if c["text_hash"] not in skip
    }
    embed.save_cache(params.paths.embeddings, params.embedding.model, vectors)


def test_run_writes_aligned_index_chunks_and_manifest(params):
    chunks = [chunk("a", 0, "un"), chunk("a", 1, "deux"), chunk("b", 0, "trois")]
    setup_pipeline_outputs(params, chunks)

    manifest = build_index.run()

    directory = params.paths.index_dir
    index = faiss.read_index(str(directory / build_index.INDEX_FILE))
    records = read_json(directory / build_index.CHUNKS_FILE)
    assert index.ntotal == len(records) == 3
    # La ligne i de l'index est le vecteur du chunk i.
    np.testing.assert_array_equal(index.reconstruct(2), [2.0, 1.0, 0.0])
    assert [r["text"] for r in records] == ["un", "deux", "trois"]
    assert all("text_for_embedding" not in r and "text_hash" not in r for r in records)
    assert manifest == {"embedding_model": params.embedding.model, "dim": 3, "n_vectors": 3, "n_events": 2}
    assert read_json(directory / build_index.MANIFEST_FILE) == manifest
    assert read_json(params.paths.index_metrics) == manifest


def test_shared_texts_reuse_the_same_vector(params):
    chunks = [chunk("a", 0, "même"), chunk("b", 0, "même")]
    setup_pipeline_outputs(params, chunks)
    build_index.run()
    index = faiss.read_index(str(params.paths.index_dir / build_index.INDEX_FILE))
    np.testing.assert_array_equal(index.reconstruct(0), index.reconstruct(1))


def test_missing_embedding_fails(params):
    chunks = [chunk("a", 0, "un"), chunk("b", 0, "deux")]
    setup_pipeline_outputs(params, chunks, skip={chunks[1]["text_hash"]})
    with pytest.raises(RuntimeError, match="1 chunk"):
        build_index.run()


def test_cache_from_another_model_is_rejected(params, monkeypatch):
    chunks = [chunk("a", 0, "un")]
    setup_pipeline_outputs(params, chunks)
    other = params.model_copy(update={"embedding": params.embedding.model_copy(update={"model": "autre-modele"})})
    monkeypatch.setattr(build_index, "get_params", lambda: other)
    with pytest.raises(RuntimeError, match="sans embedding"):
        build_index.run()


def test_empty_dataset_fails(params):
    setup_pipeline_outputs(params, [])
    with pytest.raises(RuntimeError, match="aucun chunk"):
        build_index.run()
