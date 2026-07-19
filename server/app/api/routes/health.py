"""
api/routes/health.py — Health check endpoint with Pydantic response model.
"""

import httpx
from config import settings
from fastapi import APIRouter
from qdrant_client import QdrantClient

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    qdrant_status = "unknown"
    ollama_status = "unknown"
    ollama_models = None

    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=3)
        client.get_collections()
        qdrant_status = "ok"
    except Exception as e:
        qdrant_status = f"error: {e}"

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            ollama_status = "ok"
            ollama_models = models
    except Exception as e:
        ollama_status = f"error: {e}"

    return HealthResponse(api="ok", qdrant=qdrant_status, ollama=ollama_status, ollama_models=ollama_models)
