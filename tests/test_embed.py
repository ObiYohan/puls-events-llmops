from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from src.pipeline import embed
from src.pipeline.utils import read_json, text_hash, write_json


class FakeEmbeddings:
    """Imite client.embeddings de Mistral : vecteur = [longueur du texte, 1.0]."""

    def __init__(self, errors=()):
        self.calls = []
        self.errors = list(errors)

    def create(self, model, inputs):
        self.calls.append(list(inputs))
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        # Ordre volontairement inversé : le code doit trier par index.
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(inputs)][::-1]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=10 * len(inputs)))


def fake_client(errors=()):
    return SimpleNamespace(embeddings=FakeEmbeddings(errors))


class StatusError(Exception):
    """Erreur portant un code HTTP, comme mistralai.models.SDKError (via raw_response)."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.raw_response = SimpleNamespace(status_code=status_code)


def make_chunk(text):
    return {"text_for_embedding": text, "text_hash": text_hash(text)}


@pytest.fixture
def params(isolated_params, monkeypatch):
    monkeypatch.setattr(embed, "get_params", lambda: isolated_params)
    monkeypatch.setattr(embed.time, "sleep", lambda _: None)
    return isolated_params


def write_chunks(params, texts):
    write_json(params.paths.chunks_dataset, [make_chunk(t) for t in texts])


def cached_texts(params):
    keys = read_json(params.paths.embeddings / embed.KEYS_FILE)
    return keys, np.load(params.paths.embeddings / embed.VECTORS_FILE)


# --- make_batches ------------------------------------------------------------

def test_make_batches_respects_token_limit():
    items = [("a", 40), ("b", 40), ("c", 40), ("d", 150), ("e", 10)]
    assert embed.make_batches(items, max_tokens=100) == [["a", "b"], ["c"], ["d"], ["e"]]


def test_make_batches_empty():
    assert embed.make_batches([], max_tokens=100) == []


# --- is_retryable ------------------------------------------------------------

@pytest.mark.parametrize(("error", "expected"), [
    (StatusError(429), True),
    (StatusError(503), True),
    (StatusError(401), False),
    (StatusError(400), False),
    (httpx.ConnectError("down"), True),
    (ValueError("bug"), False),
])
def test_is_retryable(error, expected):
    assert embed.is_retryable(error) is expected


# --- embed_batch -------------------------------------------------------------

def test_embed_batch_retries_with_exponential_backoff(params, monkeypatch):
    delays = []
    monkeypatch.setattr(embed.time, "sleep", delays.append)
    client = fake_client(errors=[StatusError(429), httpx.ReadTimeout("slow"), None])

    vectors, tokens = embed.embed_batch(client, ["ab", "abc"], params.embedding)

    assert vectors == [[2.0, 1.0], [3.0, 1.0]]  # remis dans l'ordre des textes
    assert tokens == 20
    base = params.embedding.retry_base_seconds
    assert delays == [base, base * 2]


def test_embed_batch_does_not_retry_auth_error(params):
    client = fake_client(errors=[StatusError(401)])
    with pytest.raises(StatusError):
        embed.embed_batch(client, ["a"], params.embedding)
    assert len(client.embeddings.calls) == 1


def test_embed_batch_gives_up_after_max_retries(params):
    max_retries = params.embedding.max_retries
    client = fake_client(errors=[StatusError(429)] * (max_retries + 1))
    with pytest.raises(StatusError):
        embed.embed_batch(client, ["a"], params.embedding)
    assert len(client.embeddings.calls) == max_retries + 1


# --- run : incrémental ---------------------------------------------------------

def test_first_run_embeds_everything(params):
    write_chunks(params, ["a", "bb", "ccc"])
    client = fake_client()

    stats = embed.run(client)

    assert stats["n_embedded"] == 3 and stats["n_reused"] == 0
    assert stats["prompt_tokens"] == 30
    keys, vectors = cached_texts(params)
    assert keys == sorted(text_hash(t) for t in ["a", "bb", "ccc"])
    assert vectors.dtype == np.float32 and vectors.shape == (3, 2)
    assert read_json(params.paths.embed_metrics) == stats
    assert read_json(params.paths.embeddings / embed.META_FILE)["model"] == params.embedding.model


def test_second_run_reuses_cache_without_api_call(params):
    write_chunks(params, ["a", "bb"])
    embed.run(fake_client())

    client = fake_client()
    stats = embed.run(client)

    assert client.embeddings.calls == []
    assert stats["n_reused"] == 2 and stats["n_embedded"] == 0


def test_only_new_texts_are_embedded_and_removed_ones_are_pruned(params):
    write_chunks(params, ["a", "bb"])
    embed.run(fake_client())

    write_chunks(params, ["bb", "new"])
    client = fake_client()
    stats = embed.run(client)

    assert client.embeddings.calls == [["new"]]
    assert stats["n_reused"] == 1 and stats["n_embedded"] == 1
    keys, _ = cached_texts(params)
    assert keys == sorted([text_hash("bb"), text_hash("new")])


def test_identical_texts_share_one_embedding(params):
    write_chunks(params, ["same", "same"])
    client = fake_client()
    stats = embed.run(client)
    assert client.embeddings.calls == [["same"]]
    assert stats["n_chunks"] == 2 and stats["n_unique_texts"] == 1


def test_model_change_invalidates_cache(params, monkeypatch):
    write_chunks(params, ["a"])
    embed.run(fake_client())

    other = params.model_copy(update={"embedding": params.embedding.model_copy(update={"model": "other-embed"})})
    monkeypatch.setattr(embed, "get_params", lambda: other)
    client = fake_client()
    stats = embed.run(client)

    assert client.embeddings.calls == [["a"]]
    assert stats["n_reused"] == 0


def test_partial_progress_is_saved_when_a_batch_fails(params, monkeypatch):
    small = params.model_copy(update={"embedding": params.embedding.model_copy(update={"max_tokens_per_batch": 1})})
    monkeypatch.setattr(embed, "get_params", lambda: small)
    write_chunks(small, ["a", "b"])  # 1 token chacun → 2 lots
    client = fake_client(errors=[None, StatusError(401)])

    with pytest.raises(StatusError):
        embed.run(client)

    keys, _ = cached_texts(small)
    assert len(keys) == 1  # le premier lot, déjà payé, est conservé
    assert not small.paths.embed_metrics.exists()

    retry_client = fake_client()
    stats = embed.run(retry_client)
    assert stats["n_reused"] == 1 and stats["n_embedded"] == 1


def test_missing_api_key_fails_only_if_something_to_embed(params, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    write_chunks(params, ["a"])
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        embed.run()

    embed.run(fake_client())
    assert embed.run()["n_reused"] == 1  # tout est en cache : pas besoin de clé


def test_corrupted_cache_is_ignored(params):
    write_chunks(params, ["a"])
    embed.run(fake_client())
    write_json(params.paths.embeddings / embed.KEYS_FILE, ["x", "y"])  # 2 clés pour 1 vecteur

    client = fake_client()
    assert embed.run(client)["n_embedded"] == 1
