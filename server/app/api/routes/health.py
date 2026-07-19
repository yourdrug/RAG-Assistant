import httpx
from config import settings
from fastapi import APIRouter
from qdrant_client import QdrantClient

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    status = {"api": "ok", "qdrant": "unknown", "ollama": "unknown"}

    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=3)
        client.get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            status["ollama"] = "ok"
            status["ollama_models"] = models
    except Exception as e:
        status["ollama"] = f"error: {e}"

    return status
