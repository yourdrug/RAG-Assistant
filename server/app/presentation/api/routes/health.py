"""Health check endpoint."""

from __future__ import annotations

import httpx
from config import settings
from fastapi import APIRouter
from qdrant_client import QdrantClient

from presentation.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


async def get_ollama_models() -> list[str] | None:
    """Fetch list of available Ollama models. Returns None on error."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def get_qdrant_status(timeout: int = 3) -> str:
    """Check Qdrant connectivity. Returns 'ok' or 'error: ...'."""
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=timeout)
        client.get_collections()
        return "ok"
    except Exception as e:
        return f"error: {e}"


@router.get("/health", response_model=HealthResponse)
async def health():
    qdrant_status = get_qdrant_status()
    ollama_models = await get_ollama_models()
    ollama_status = "ok" if ollama_models is not None else "error: unable to reach Ollama"

    return HealthResponse(api="ok", qdrant=qdrant_status, ollama=ollama_status, ollama_models=ollama_models)
