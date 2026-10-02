"""Health-check endpoint aggregating Postgres, Qdrant, and Ollama status."""

from __future__ import annotations

from application.services.health_service import HealthService
from fastapi import APIRouter, Depends, Response, status
from domain.value_objects.health_status import HealthStatus

from presentation.api.dependencies import create_health_service
from presentation.api.schemas import HealthCheck, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(health_service: HealthService = Depends(create_health_service)):
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


@router.get("/live", include_in_schema=False)
async def live() -> dict[str, str]:
    """Process liveness; independent of external dependencies."""
    return {"status": HealthStatus.OK.value}


@router.get("/ready", response_model=HealthResponse, include_in_schema=False)
async def ready(response: Response, health_service: HealthService = Depends(create_health_service)):
    result = await health(health_service)
    if result.status != HealthStatus.HEALTHY.value:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
