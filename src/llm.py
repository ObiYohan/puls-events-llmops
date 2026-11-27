"""Client LLM unique du projet : Azure OpenAI via l'endpoint v1 (compatible OpenAI).

Les noms de modèles de ``params.yaml`` sont aussi les noms des déploiements
Azure (convention : un déploiement porte le nom de son modèle).

Secrets (variables d'environnement, sans valeur par défaut) :

- ``AZURE_OPENAI_ENDPOINT`` : URL de la ressource (``https://<ressource>.openai.azure.com``,
  ``https://<ressource>.services.ai.azure.com``) ou du projet Foundry ;
- ``AZURE_OPENAI_API_KEY`` : clé de la ressource.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from openai import APIConnectionError, OpenAI

from src.config import get_secret

logger = logging.getLogger(__name__)

T = TypeVar("T")

ENDPOINT_VAR = "AZURE_OPENAI_ENDPOINT"
API_KEY_VAR = "AZURE_OPENAI_API_KEY"

V1_PATH = "/openai/v1"
PROJECT_PATH = "/api/projects/"


def v1_base_url(endpoint: str) -> str:
    """URL de l'endpoint v1 de la ressource, quelle que soit la forme copiée du portail.

    Accepte l'URL de la ressource (``…openai.azure.com`` ou ``…services.ai.azure.com``),
    avec ou sans ``/openai/v1``, ou l'URL d'un projet Foundry
    (``…/api/projects/<projet>``), ramenée à celle de sa ressource : l'authentification
    par clé se fait au niveau de la ressource.
    """
    url = endpoint.strip().rstrip("/")
    url = url.split(PROJECT_PATH, 1)[0]
    if not url.endswith(V1_PATH):
        url += V1_PATH
    return url + "/"


def make_client(max_retries: int = 2) -> OpenAI:
    """Construit le client ; échoue explicitement si un secret manque.

    ``max_retries=0`` désactive les retries du SDK, pour les appelants qui
    gèrent eux-mêmes le backoff (pipeline d'embeddings).
    """
    endpoint, api_key = get_secret(ENDPOINT_VAR), get_secret(API_KEY_VAR)
    missing = [name for name, value in ((ENDPOINT_VAR, endpoint), (API_KEY_VAR, api_key)) if value is None]
    if missing:
        raise RuntimeError(f"secret(s) absent(s) : {', '.join(missing)}")
    return OpenAI(api_key=api_key, base_url=v1_base_url(endpoint), max_retries=max_retries)


def status_code(exc: Exception) -> int | None:
    """Code HTTP porté par une erreur d'API, directement ou via sa réponse."""
    for obj in (exc, getattr(exc, "response", None), getattr(exc, "raw_response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def is_retryable(exc: Exception) -> bool:
    """429 (quota), 5xx et erreurs réseau sont retentés ; le reste (401, 400…) ne l'est pas."""
    code = status_code(exc)
    if code is not None:
        return code == 429 or code >= 500
    return isinstance(exc, APIConnectionError)


def with_retry(call: Callable[[], T], *, max_retries: int, base_seconds: float, name: str = "llm_call") -> T:
    """Exécute ``call`` avec backoff exponentiel (base, 2×base, 4×base…) sur les erreurs retentables."""
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:
            if not is_retryable(exc) or attempt == max_retries:
                raise
            delay = base_seconds * 2**attempt
            logger.warning(
                "llm_retry",
                extra={"call": name, "attempt": attempt + 1, "status_code": status_code(exc), "delay_seconds": delay},
            )
            time.sleep(delay)
    raise AssertionError("inatteignable")  # pragma: no cover
