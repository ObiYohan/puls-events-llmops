from types import SimpleNamespace

import httpx
import openai
import pytest

from src import llm


def _status_error(code):
    request = httpx.Request("POST", "https://test")
    return openai.APIStatusError("erreur", response=httpx.Response(code, request=request), body=None)


@pytest.mark.parametrize("endpoint", [
    "https://res.openai.azure.com",
    "https://res.openai.azure.com/",
    "https://res.openai.azure.com/openai/v1",
    " https://res.openai.azure.com/openai/v1/ ",
])
def test_v1_base_url_is_normalized(endpoint):
    assert llm.v1_base_url(endpoint) == "https://res.openai.azure.com/openai/v1/"


@pytest.mark.parametrize("endpoint", [
    "https://res.services.ai.azure.com/api/projects/puls-events",
    "https://res.services.ai.azure.com/api/projects/puls-events/",
    "https://res.services.ai.azure.com/",
])
def test_v1_base_url_accepts_foundry_project_and_resource(endpoint):
    assert llm.v1_base_url(endpoint) == "https://res.services.ai.azure.com/openai/v1/"


def test_make_client_uses_v1_endpoint_and_key():
    client = llm.make_client(max_retries=0)
    assert str(client.base_url) == "https://test-resource.openai.azure.com/openai/v1/"
    assert client.api_key == "test_azure_key_12345"
    assert client.max_retries == 0


@pytest.mark.parametrize("missing", [llm.ENDPOINT_VAR, llm.API_KEY_VAR])
def test_make_client_fails_without_secret(monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        llm.make_client()


@pytest.mark.parametrize(("error", "expected"), [
    (_status_error(429), True),
    (_status_error(503), True),
    (_status_error(401), False),
    (_status_error(400), False),
    (openai.APIConnectionError(request=httpx.Request("POST", "https://test")), True),
    (openai.APITimeoutError(request=httpx.Request("POST", "https://test")), True),
    (ValueError("bug"), False),
])
def test_is_retryable(error, expected):
    assert llm.is_retryable(error) is expected


def test_status_code_reads_raw_response():
    error = Exception("x")
    error.raw_response = SimpleNamespace(status_code=502)
    assert llm.status_code(error) == 502
