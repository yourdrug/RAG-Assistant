"""Health-check endpoint aggregating Postgres, Qdrant, and Ollama status."""

from __future__ import annotations

from application.services.health_service import HealthService
from fastapi import APIRouter, Depends

from presentation.api.dependencies import get_health_service
from presentation.api.schemas import HealthCheck, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(health_service: HealthService = Depends(get_health_service)):
    result = await health_service.check()
    return HealthResponse(
        status=result.status,
        version=result.version,
        uptime_seconds=result.uptime_seconds,
        llm_provider=result.llm_provider,
        checks={
            k: HealthCheck(status=v.status, latency_ms=v.latency_ms, models=v.models)
            for k, v in result.checks.items()
        },
        background_jobs=result.background_jobs,
    )
