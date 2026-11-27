"""API FastAPI du chatbot Puls-Events.

Au démarrage, l'API charge l'index « champion » du registre MLflow
(voir ``src/serving.py``) ; ``/reload`` recharge le champion courant après une
promotion, sans redéploiement.

Sécurité : clé d'administration sans valeur par défaut (``ADMIN_API_KEY`` ;
absente → endpoints protégés désactivés), CORS limité aux origines de
``ALLOWED_ORIGINS``, erreurs 500 génériques (le détail part dans les logs),
question jamais journalisée en clair (hash et longueur seulement).
"""

import hashlib
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chatbot import EventChatbot  # noqa: E402
from src.config import get_params, get_secret  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.serving import ServedIndex, load_served_index  # noqa: E402

logger = logging.getLogger("api")

ADMIN_KEY_VAR = "ADMIN_API_KEY"
ORIGINS_VAR = "ALLOWED_ORIGINS"
MAX_QUERY_LENGTH = 1000
GENERIC_ERROR = "Erreur interne : réessayez plus tard."

chatbot_instance: EventChatbot | None = None
served_index: ServedIndex | None = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Vérifie la clé d'administration des endpoints protégés."""
    expected = get_secret(ADMIN_KEY_VAR)
    if expected is None:
        raise HTTPException(status_code=503, detail="Endpoint d'administration désactivé")
    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Clé d'API invalide ou absente")
    return api_key


def allowed_origins() -> list[str]:
    """Origines CORS autorisées (liste séparée par des virgules) ; aucune par défaut."""
    raw = get_secret(ORIGINS_VAR) or ""
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def load_chatbot() -> tuple[EventChatbot, ServedIndex]:
    """Charge l'index servi (champion MLflow ou index local) et le chatbot associé."""
    params = get_params()
    served = load_served_index(params)
    chatbot = EventChatbot(index_path=served.index_path, chunks_path=served.chunks_path, params=params)
    return chatbot, served


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot_instance, served_index
    setup_logging()
    try:
        chatbot_instance, served_index = load_chatbot()
    except Exception:
        logger.exception("chatbot_load_failed")
        chatbot_instance, served_index = None, None
    yield
    chatbot_instance, served_index = None, None


app = FastAPI(
    title="Puls-Events — API du chatbot",
    description="Recommandation d'événements culturels en région PACA (RAG)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# --- Modèles ---------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="Question de l'utilisateur", min_length=1, max_length=MAX_QUERY_LENGTH)
    k: int = Field(default_factory=lambda: get_params().retrieval.k, description="Nombre d'événements", ge=1, le=20)
    show_sources: bool = Field(default=True, description="Inclure les sources utilisées")


class Source(BaseModel):
    uid: str
    chunk_id: int
    text: str
    distance: float
    firstdate_begin: Optional[str] = None
    lastdate_end: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Source]] = None
    query: str
    k: int
    index_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    chatbot_loaded: bool
    index_size: Optional[int] = None
    index_source: Optional[str] = None
    index_version: Optional[str] = None


class ReloadResponse(BaseModel):
    status: str
    message: str
    index_size: int
    total_events: int
    total_chunks: int
    index_source: str
    index_version: Optional[str] = None


def _require_chatbot() -> EventChatbot:
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot non initialisé")
    return chatbot_instance


def _version() -> Optional[str]:
    return served_index.version if served_index else None


# --- Endpoints -------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """État de l'API et version de l'index servi."""
    if chatbot_instance is None:
        return HealthResponse(status="unhealthy", chatbot_loaded=False)
    return HealthResponse(
        status="healthy",
        chatbot_loaded=True,
        index_size=chatbot_instance.index.ntotal,
        index_source=served_index.source if served_index else None,
        index_version=_version(),
    )


@app.post("/ask", response_model=QueryResponse, tags=["Chatbot"])
def ask(request: QueryRequest):
    """Pose une question au chatbot (fonction synchrone : exécutée hors de la boucle d'événements)."""
    chatbot = _require_chatbot()
    started = time.perf_counter()
    question_hash = hashlib.sha256(request.query.encode("utf-8")).hexdigest()[:16]
    try:
        result = chatbot.chat(request.query, k=request.k)
        sources = None
        if request.show_sources:
            sources = [
                Source(
                    uid=str(row["uid"]),
                    chunk_id=int(row["chunk_id"]),
                    text=row["text"][:500] + "..." if len(row["text"]) > 500 else row["text"],
                    distance=float(result["distances"][idx]),
                    firstdate_begin=str(row.get("firstdate_begin")) if "firstdate_begin" in row else None,
                    lastdate_end=str(row.get("lastdate_end")) if "lastdate_end" in row else None,
                )
                for idx, (_, row) in enumerate(result["sources"].iterrows())
            ]
    except Exception:
        logger.exception("ask_failed", extra={"question_hash": question_hash, "index_version": _version()})
        raise HTTPException(status_code=500, detail=GENERIC_ERROR)

    logger.info("ask_done", extra={
        "question_hash": question_hash,
        "question_length": len(request.query),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "n_sources": len(result["sources"]),
        "index_version": _version(),
    })
    return QueryResponse(
        response=result["response"], sources=sources, query=request.query, k=request.k, index_version=_version()
    )


@app.post("/reload", response_model=ReloadResponse, tags=["System"])
def reload_index(api_key: str = Depends(verify_api_key)):
    """Recharge l'index champion (après une promotion) ; clé d'administration requise.

    L'index est construit et promu par le pipeline (DVC + MLflow), pas par l'API.
    """
    global chatbot_instance, served_index
    try:
        new_chatbot, new_index = load_chatbot()
    except Exception:
        logger.exception("reload_failed")
        raise HTTPException(status_code=500, detail="Échec du rechargement de l'index")
    chatbot_instance, served_index = new_chatbot, new_index
    logger.info("reload_done", extra={"index_version": new_index.version, "index_source": new_index.source})
    return ReloadResponse(
        status="success",
        message="Index rechargé",
        index_size=new_chatbot.index.ntotal,
        total_events=len(new_chatbot.chunks_df["uid"].unique()),
        total_chunks=len(new_chatbot.chunks_df),
        index_source=new_index.source,
        index_version=new_index.version,
    )


@app.get("/stats", tags=["System"])
async def get_stats():
    """Statistiques de l'index servi."""
    chatbot = _require_chatbot()
    try:
        return {
            "total_events": len(chatbot.chunks_df["uid"].unique()),
            "total_chunks": len(chatbot.chunks_df),
            "index_size": chatbot.index.ntotal,
            "embedding_dimension": chatbot.index.d,
            "index_version": _version(),
            "embedding_model": served_index.manifest.get("embedding_model") if served_index else None,
        }
    except Exception:
        logger.exception("stats_failed")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR)


@app.get("/", tags=["System"])
async def root():
    return {"message": "Puls-Events — API du chatbot", "version": app.version, "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
