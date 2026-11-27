from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main_api as main_api


@pytest.fixture
def client(monkeypatch):
    # Sans bloc `with`, le lifespan (chargement de l'index réel) n'est pas exécuté.
    monkeypatch.setattr(main_api, "chatbot_instance", None)
    return TestClient(main_api.app)


def fake_chatbot():
    return SimpleNamespace(
        index=SimpleNamespace(ntotal=3, d=4),
        chunks_df=pd.DataFrame({"uid": ["a", "a", "b"]}),
    )


def test_reload_is_disabled_without_admin_key(client, monkeypatch):
    monkeypatch.delenv(main_api.ADMIN_KEY_VAR, raising=False)
    response = client.post("/reload", headers={"X-API-Key": "nimporte"})
    assert response.status_code == 503


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "mauvaise"}])
def test_reload_rejects_missing_or_wrong_key(client, monkeypatch, headers):
    monkeypatch.setenv(main_api.ADMIN_KEY_VAR, "bonne-cle")
    response = client.post("/reload", headers=headers)
    assert response.status_code == 403


def test_reload_with_valid_key_reloads_index(client, monkeypatch):
    monkeypatch.setenv(main_api.ADMIN_KEY_VAR, "bonne-cle")
    monkeypatch.setattr(main_api, "EventChatbot", fake_chatbot)

    response = client.post("/reload", headers={"X-API-Key": "bonne-cle"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "success", "message": "Index rechargé",
        "index_size": 3, "total_events": 2, "total_chunks": 3,
    }


def test_reload_failure_returns_generic_error(client, monkeypatch):
    monkeypatch.setenv(main_api.ADMIN_KEY_VAR, "bonne-cle")

    def broken():
        raise FileNotFoundError("C:/chemin/secret/faiss.index")

    monkeypatch.setattr(main_api, "EventChatbot", broken)
    response = client.post("/reload", headers={"X-API-Key": "bonne-cle"})

    assert response.status_code == 500
    assert "secret" not in response.text


def test_health_reports_unloaded_chatbot(client):
    assert client.get("/health").json() == {"status": "unhealthy", "chatbot_loaded": False, "index_size": None}
