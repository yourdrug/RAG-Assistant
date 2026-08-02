"""Health check endpoint."""

from __future__ import annotations

import time

import httpx
from config import settings
from fastapi import APIRouter
from infrastructure.database.database import database
from qdrant_client import QdrantClient
from sqlalchemy import text

from presentation.api.schemas import HealthCheck, HealthResponse

router = APIRouter(tags=["health"])


async def _check_ollama() -> HealthCheck:
    """Check Ollama connectivity with latency measurement."""
    try:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            models = [m["name"] for m in r.json().get("models", [])]
            return HealthCheck(status="ok", latency_ms=latency_ms, models=models)
    except Exception as e:
        return HealthCheck(status=f"error: {e}")


def _check_qdrant() -> HealthCheck:
    """Check Qdrant connectivity with latency measurement."""
    try:
        t0 = time.perf_counter()
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=3)
        client.get_collections()
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return HealthCheck(status="ok", latency_ms=latency_ms)
    except Exception as e:
        return HealthCheck(status=f"error: {e}")


async def _check_postgres() -> HealthCheck:
    """Check PostgreSQL connectivity with latency measurement."""
    try:
        t0 = time.perf_counter()
        session = database.get_write_session()
        async with session:
            await session.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return HealthCheck(status="ok", latency_ms=latency_ms)
    except Exception as e:
        return HealthCheck(status=f"error: {e}")


async def _count_active_jobs() -> int:
    """Count currently running/pending background jobs."""
    from presentation.api.dependencies import _uow_factory

    try:
        async with _uow_factory.create() as uow:
            return await uow.background_jobs.count_active()
    except Exception:
        return 0


@router.get("/health", response_model=HealthResponse)
async def health():
    qdrant = _check_qdrant()
    ollama = await _check_ollama()
    postgres = await _check_postgres()
    active_jobs = await _count_active_jobs()

    overall = "healthy"
    if any(c.status.startswith("error") for c in [qdrant, ollama, postgres]):
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version=settings.version,
        uptime_seconds=settings.uptime_seconds,
        checks={
            "api": HealthCheck(status="ok"),
            "qdrant": qdrant,
            "ollama": ollama,
            "postgres": postgres,
        },
        background_jobs={"running": active_jobs},
    )


# ---------------------------------------------------------------------------
# Helpers (used by admin_config.py)
# ---------------------------------------------------------------------------


async def get_ollama_models() -> list[str]:
    """Return list of Ollama model names."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def get_qdrant_status(timeout: int = 3) -> str:
    """Return 'ok' or error string for Qdrant connectivity."""
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=timeout)
        client.get_collections()
        return "ok"
    except Exception as e:
        return f"error: {e}"
