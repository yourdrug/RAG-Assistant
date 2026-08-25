"""Health-check endpoints — aggregating Postgres, Qdrant, and Ollama status.

Provides three levels of health checking for Kubernetes:

- ``/health``       — full health report (legacy, same as before)
- ``/health/live``  — liveness probe: process is alive
- ``/health/ready`` — readiness probe: can accept production traffic
"""

from __future__ import annotations

import time

from application.services.health_service import HealthService
from fastapi import APIRouter, Depends
from infrastructure.database.database import database
from infrastructure.persistence.redis_client import redis_client

from presentation.api.dependencies import create_health_service
from presentation.api.schemas import HealthCheck, HealthResponse

router = APIRouter(tags=["health"])

_start_time: float = time.time()


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


@router.get("/health/live")
async def health_live():
    """Liveness probe — confirms the process is alive and responsive.

    This check intentionally does NOT verify external dependencies
    (Postgres, Qdrant, Redis) to avoid restarting a healthy process
    because of a transient downstream issue.
    """
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@router.get("/health/ready")
async def health_ready(health_service: HealthService = Depends(create_health_service)):
    """Readiness probe — confirms the pod can accept production traffic.

    Checks Postgres and Redis connectivity.  A failure here removes the
    pod from the Service endpoints without killing the process.
    """
    checks: dict[str, str] = {}

    # Postgres
    try:
        from sqlalchemy import text

        session = database.get_write_session()
        async with session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    # Redis
    try:
        redis = redis_client.async_redis
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }
