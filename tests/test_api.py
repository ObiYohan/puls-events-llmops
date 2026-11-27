from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main_api as main_api
from src.serving import ServedIndex


@pytest.fixture
def client(monkeypatch):
    # Sans bloc `with`, le lifespan (téléchargement du champion) n'est pas exécuté.
    monkeypatch.setattr(main_api, "chatbot_instance", None)
    monkeypatch.setattr(main_api, "served_index", None)
    return TestClient(main_api.app)


def fake_served(version="2"):
    return ServedIndex(Path("x"), "mlflow", version, "run123", {"embedding_model": "text-embedding-3-small"})


class FakeChatbot:
    def __init__(self, fail=False):
        self.fail = fail
        self.index = SimpleNamespace(ntotal=3, d=4)
        self.chunks_df = pd.DataFrame({"uid": ["a", "a", "b"]})

    def chat(self, query, k=None):
        if self.fail:
            raise RuntimeError("C:/chemin/secret Azure indisponible")
        sources = pd.DataFrame({"uid": ["a"], "chunk_id": [0], "text": ["Concert"], "lastdate_end": ["2099-01-01"]})
        return {"response": "Voici un concert.", "sources": sources, "distances": [0.12]}


def install(monkeypatch, chatbot=None, served=None):
    monkeypatch.setattr(main_api, "chatbot_instance", chatbot or FakeChatbot())
    monkeypatch.setattr(main_api, "served_index", served or fake_served())


# --- /reload ---------------------------------------------------------------------

def test_reload_is_disabled_without_admin_key(client, monkeypatch):
    monkeypatch.delenv(main_api.ADMIN_KEY_VAR, raising=False)
    assert client.post("/reload", headers={"X-API-Key": "nimporte"}).status_code == 503


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "mauvaise"}])
def test_reload_rejects_missing_or_wrong_key(client, monkeypatch, headers):
    monkeypatch.setenv(main_api.ADMIN_KEY_VAR, "bonne-cle")
    assert client.post("/reload", headers=headers).status_code == 403


def test_reload_with_valid_key_loads_current_champion(client, monkeypatch):
    monkeypatch.setenv(main_api.ADMIN_KEY_VAR, "bonne-cle")
    monkeypatch.setattr(main_api, "load_chatbot", lambda: (FakeChatbot(), fake_served("7")))

    response = client.post("/reload", headers={"X-API-Key": "bonne-cle"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "success", "message": "Index rechargé", "index_size": 3, "total_events": 2,
        "total_chunks": 3, "index_source": "mlflow", "index_version": "7",
    }
    assert main_api.served_index.version == "7"


def test_reload_failure_keeps_current_index_and_hides_details(client, monkeypatch):
    monkeypatch.setenv(main_api.ADMIN_KEY_VAR, "bonne-cle")
    install(monkeypatch)

    def broken():
        raise FileNotFoundError("C:/chemin/secret/faiss.index")

    monkeypatch.setattr(main_api, "load_chatbot", broken)
    response = client.post("/reload", headers={"X-API-Key": "bonne-cle"})

    assert response.status_code == 500 and "secret" not in response.text
    assert main_api.served_index.version == "2"  # l'index précédent reste servi


# --- /health, /ask, /stats ------------------------------------------------------

def test_health_reports_unloaded_chatbot(client):
    assert client.get("/health").json()["status"] == "unhealthy"


def test_health_reports_served_version(client, monkeypatch):
    install(monkeypatch)
    body = client.get("/health").json()
    assert body == {"status": "healthy", "chatbot_loaded": True, "index_size": 3,
                    "index_source": "mlflow", "index_version": "2"}


def test_ask_returns_answer_sources_and_version(client, monkeypatch):
    install(monkeypatch)
    body = client.post("/ask", json={"query": "Un concert ?"}).json()
    assert body["response"] == "Voici un concert."
    assert body["sources"][0]["uid"] == "a" and body["index_version"] == "2"


def test_ask_logs_hash_but_never_the_question(client, monkeypatch, caplog):
    install(monkeypatch)
    caplog.set_level("INFO")
    client.post("/ask", json={"query": "question de Jeanne Dupont"})
    record = next(r for r in caplog.records if r.getMessage() == "ask_done")
    assert len(record.question_hash) == 16 and record.question_length == 25
    assert "Jeanne" not in caplog.text


def test_ask_error_is_generic(client, monkeypatch):
    install(monkeypatch, chatbot=FakeChatbot(fail=True))
    response = client.post("/ask", json={"query": "Un concert ?"})
    assert response.status_code == 500
    assert response.json()["detail"] == main_api.GENERIC_ERROR
    assert "secret" not in response.text


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "x" * (main_api.MAX_QUERY_LENGTH + 1)}, {"query": "ok", "k": 99}])
def test_ask_validates_input(client, monkeypatch, payload):
    install(monkeypatch)
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_without_chatbot_is_unavailable(client):
    assert client.post("/ask", json={"query": "Un concert ?"}).status_code == 503


def test_stats_include_served_version(client, monkeypatch):
    install(monkeypatch)
    body = client.get("/stats").json()
    assert body["index_version"] == "2" and body["embedding_model"] == "text-embedding-3-small"


# --- CORS ------------------------------------------------------------------------

def test_allowed_origins_default_to_none(monkeypatch):
    monkeypatch.delenv(main_api.ORIGINS_VAR, raising=False)
    assert main_api.allowed_origins() == []


def test_allowed_origins_are_parsed(monkeypatch):
    monkeypatch.setenv(main_api.ORIGINS_VAR, "https://a.example, https://b.example ,")
    assert main_api.allowed_origins() == ["https://a.example", "https://b.example"]
