"""Health + admin probe adapters — wraps infrastructure checks behind ports.

Consolidates SystemHealthProbe (connectivity checks) with OllamaProbe/QdrantInfo
(admin info providers) into a single module.  The admin/ directory re-exports
these symbols for backward compatibility.
"""

from __future__ import annotations

import asyncio
import time

import httpx
from application.ports.health import HealthCheckResult
from config import settings
from domain.value_objects.health_status import HealthStatus
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client
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
                r = await client.get(
                    f"{settings.openrouter_base_url}/models",
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

    async def check_qdrant(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            await asyncio.to_thread(self._check_qdrant_sync)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return HealthCheckResult(status=HealthStatus.OK.value, latency_ms=latency_ms)
        except Exception as e:
            return HealthCheckResult(status=f"error: {e}")

    def _check_qdrant_sync(self) -> None:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=3)
        client.get_collections()

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

    async def check_redis(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            await redis_client.async_redis.ping()
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return HealthCheckResult(status=HealthStatus.OK.value, latency_ms=latency_ms)
        except Exception as e:
            return HealthCheckResult(status=f"error: {e}")


class OllamaProbe:
    """Ollama model listing — delegates connectivity check to SystemHealthProbe."""

    def __init__(self, health_probe: SystemHealthProbe | None = None) -> None:
        self._health = health_probe

    async def get_models(self) -> list[str]:
        if self._health is not None:
            result = await self._health.check_ollama()
            return result.models or []
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{settings.ollama_base_url}/api/tags")
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []


class QdrantInfo:
    """Qdrant collection info — delegates status check to SystemHealthProbe."""

    def __init__(self, health_probe: SystemHealthProbe | None = None) -> None:
        self._health = health_probe
        self._client: QdrantClient | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
        return self._client

    async def get_status(self) -> str:
        if self._health is not None:
            result = await self._health.check_qdrant()
            return result.status
        try:
            client = self._get_client()
            await asyncio.to_thread(client.get_collections)
            return HealthStatus.OK.value
        except Exception as e:
            return f"error: {e}"

    async def get_collections(self) -> list[dict]:
        try:
            client = self._get_client()
            return await asyncio.to_thread(self._fetch_collections, client)
        except Exception:
            return []

    @staticmethod
    def _fetch_collections(client: QdrantClient) -> list[dict]:
        col_list = client.get_collections()
        result = []
        for col in col_list.collections:
            info = client.get_collection(col.name)
            vectors_cfg = info.config.params.vectors if info.config.params.vectors else None
            hnsw_cfg = info.config.hnsw_config
            vector_size = None
            vector_distance = None
            if isinstance(vectors_cfg, dict):
                if vectors_cfg:
                    first = next(iter(vectors_cfg.values()))
                    vector_size = first.size
                    d = first.distance
                    vector_distance = str(d.value) if hasattr(d, "value") else str(d)
            elif vectors_cfg is not None:
                vector_size = vectors_cfg.size
                d = vectors_cfg.distance
                vector_distance = str(d.value) if hasattr(d, "value") else str(d)
            result.append(
                {
                    "name": col.name,
                    "points_count": info.points_count or 0,
                    "vectors_count": info.vectors_count or 0,
                    "indexed_vectors_count": info.indexed_vectors_count or 0,
                    "segments_count": info.segments_count or 0,
                    "status": str(info.status.value) if hasattr(info.status, "value") else str(info.status),
                    "optimizer_status": (
                        str(info.optimizer_status.value)
                        if hasattr(info.optimizer_status, "value")
                        else str(info.optimizer_status)
                    ),
                    "hnsw_m": hnsw_cfg.m if hnsw_cfg else None,
                    "hnsw_ef_construct": hnsw_cfg.ef_construct if hnsw_cfg else None,
                    "on_disk_payload": info.config.params.on_disk_payload if info.config.params else None,
                    "vector_size": vector_size,
                    "distance": vector_distance,
                }
            )
        return result
