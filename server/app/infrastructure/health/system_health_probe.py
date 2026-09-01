"""Health probe adapter — wraps infrastructure health checks behind ports."""

from __future__ import annotations

import time

import httpx
from application.ports.health import HealthCheckResult
from config import settings
from domain.value_objects.health_status import HealthStatus
from infrastructure.database.database import database
from qdrant_client import QdrantClient
from sqlalchemy import text


class SystemHealthProbe:
    """Adapts infrastructure connectivity checks behind HealthProbePort."""

    async def check_ollama(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{settings.ollama_base_url}/api/tags")
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                models = [m["name"] for m in r.json().get("models", [])]
                return HealthCheckResult(status=HealthStatus.OK.value, latency_ms=latency_ms, models=models)
        except Exception as e:
            return HealthCheckResult(status=f"error: {e}")

    async def check_openrouter(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.head(
                    settings.openrouter_base_url,
                    headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                )
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                if r.status_code < 400:
                    return HealthCheckResult(
                        status=HealthStatus.OK.value,
                        latency_ms=latency_ms,
                        models=[settings.openrouter_model],
                    )
                return HealthCheckResult(status=f"error: HTTP {r.status_code}")
        except Exception as e:
            return HealthCheckResult(status=f"error: {e}")

    async def check_deepinfra(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.head(
                    settings.deepinfra_base_url,
                    headers={"Authorization": f"Bearer {settings.deepinfra_api_key}"},
                )
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                if r.status_code < 400:
                    return HealthCheckResult(
                        status=HealthStatus.OK.value,
                        latency_ms=latency_ms,
                        models=[settings.deepinfra_embed_model, settings.deepinfra_rerank_model],
                    )
                return HealthCheckResult(status=f"error: HTTP {r.status_code}")
        except Exception as e:
            return HealthCheckResult(status=f"error: {e}")

    def check_qdrant(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=3)
            client.get_collections()
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return HealthCheckResult(status=HealthStatus.OK.value, latency_ms=latency_ms)
        except Exception as e:
            return HealthCheckResult(status=f"error: {e}")

    async def check_postgres(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            session = database.get_write_session()
            async with session:
                await session.execute(text("SELECT 1"))
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return HealthCheckResult(status=HealthStatus.OK.value, latency_ms=latency_ms)
        except Exception as e:
            return HealthCheckResult(status=f"error: {e}")
