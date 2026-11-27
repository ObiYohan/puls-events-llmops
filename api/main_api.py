from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, Security, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
import secrets
import sys
import os

# Add parent directory to path to import chatbot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chatbot import EventChatbot
from src.config import get_params, get_secret
from src.pipeline.build_index import CHUNKS_FILE, INDEX_FILE

# Global chatbot instance
chatbot_instance = None

# Sécurité : clé d'administration lue dans ADMIN_API_KEY, sans valeur par défaut.
# Si elle est absente, les endpoints protégés sont désactivés.
ADMIN_KEY_VAR = "ADMIN_API_KEY"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Vérifie la clé d'administration des endpoints protégés."""
    expected = get_secret(ADMIN_KEY_VAR)
    if expected is None:
        raise HTTPException(status_code=503, detail="Endpoint d'administration désactivé")
    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Clé d'API invalide ou absente")
    return api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown)"""
    # Startup
    global chatbot_instance
    try:
        chatbot_instance = EventChatbot()
        print("✅ Chatbot initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        chatbot_instance = None
    
    yield  # Application is running
    
    # Shutdown (optional cleanup)
    print("🔄 Shutting down chatbot...")
    chatbot_instance = None

app = FastAPI(
    title="Event Chatbot API",
    description="API pour recommander des événements culturels avec RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS pour une environnement de développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Question de l'utilisateur", min_length=1)
    k: int = Field(default_factory=lambda: get_params().retrieval.k, description="Nombre d'événements à rechercher", ge=1, le=20)
    show_sources: bool = Field(default=True, description="Afficher les sources utilisées")

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

class HealthResponse(BaseModel):
    status: str
    chatbot_loaded: bool
    index_size: Optional[int] = None

class ReloadResponse(BaseModel):
    status: str
    message: str
    index_size: int
    total_events: int
    total_chunks: int


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Vérifie l'état de l'API et du chatbot"""
    if chatbot_instance is None:
        return HealthResponse(
            status="unhealthy",
            chatbot_loaded=False,
            index_size=None
        )
    
    return HealthResponse(
        status="healthy",
        chatbot_loaded=True,
        index_size=chatbot_instance.index.ntotal
    )

# Main chat endpoint
@app.post("/ask", response_model=QueryResponse, tags=["Chatbot"])
async def ask(request: QueryRequest):
    """
    Pose une question au chatbot et obtient une réponse augmentée
    
    - **query**: La question de l'utilisateur
    - **k**: Nombre d'événements similaires à rechercher (1-20)
    - **show_sources**: Inclure les sources dans la réponse
    """
    if chatbot_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized. Please check server logs."
        )
    
    try:
        # Get chatbot response
        result = chatbot_instance.chat(
            request.query,
            k=request.k,
            show_sources=False
        )
        
        # Format sources if requested
        sources = None
        if request.show_sources:
            sources = []
            for idx, (_, row) in enumerate(result["sources"].iterrows()):
                sources.append(Source(
                    uid=str(row["uid"]),
                    chunk_id=int(row["chunk_id"]),
                    text=row["text"][:500] + "..." if len(row["text"]) > 500 else row["text"],
                    distance=float(result["distances"][idx]),
                    firstdate_begin=str(row.get("firstdate_begin")) if "firstdate_begin" in row else None,
                    lastdate_end=str(row.get("lastdate_end")) if "lastdate_end" in row else None
                ))
        
        return QueryResponse(
            response=result["response"],
            sources=sources,
            query=request.query,
            k=request.k
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Recharge l'index depuis le disque (après `dvc pull` ou un nouveau build)
@app.post("/reload", response_model=ReloadResponse, tags=["System"])
async def reload_index(api_key: str = Depends(verify_api_key)):
    """Recharge l'index FAISS et les chunks (clé d'administration requise).

    L'index est construit par le pipeline DVC (`dvc repro`), pas par l'API.
    """
    global chatbot_instance
    try:
        new_instance = EventChatbot()
    except Exception as e:
        print(f"❌ Échec du rechargement : {e}")
        raise HTTPException(status_code=500, detail="Échec du rechargement de l'index")
    chatbot_instance = new_instance
    return ReloadResponse(
        status="success",
        message="Index rechargé",
        index_size=chatbot_instance.index.ntotal,
        total_events=len(chatbot_instance.chunks_df["uid"].unique()),
        total_chunks=len(chatbot_instance.chunks_df),
    )

# Get chatbot statistics
@app.get("/stats", tags=["System"])
async def get_stats():
    """Obtient les statistiques du chatbot"""
    if chatbot_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized"
        )
    
    try:
        return {
            "total_events": len(chatbot_instance.chunks_df["uid"].unique()),
            "total_chunks": len(chatbot_instance.chunks_df),
            "index_size": chatbot_instance.index.ntotal,
            "embedding_dimension": chatbot_instance.index.d
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get current index status"""
    index_dir = get_params().paths.index_dir
    files = {
        "data_file_exists": (index_dir / CHUNKS_FILE).exists(),
        "index_file_exists": (index_dir / INDEX_FILE).exists(),
    }
    if chatbot_instance is None:
        return {"index_loaded": False, "total_chunks": 0, "index_size": 0, **files}

    return {
        "index_loaded": True,
        "total_chunks": len(chatbot_instance.chunks_df),
        "index_size": chatbot_instance.index.ntotal,
        **files,
    }

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Event Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)