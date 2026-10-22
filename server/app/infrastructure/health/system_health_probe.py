"""Connectivity checks and admin information behind application ports."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import closing
from typing import Any

import httpx
from application.health.ports import HealthCheckResult
from config import settings
from domain.runtime.enums import HealthStatus
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client
from qdrant_client import QdrantClient
from sqlalchemy import text


logger = logging.getLogger("default")


class HttpProbe:
    """Perform a timed HTTP request and release its client after each probe."""

    async def get(
        self, url: str, *, request_timeout: float = 3, headers: dict[str, str] | None = None
    ) -> tuple[httpx.Response, float]:
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.get(url, headers=headers)
        return response, round((time.perf_counter() - started) * 1000, 1)


class SystemHealthProbe:
    """Adapts infrastructure connectivity checks behind HealthProbePort."""

    def __init__(self) -> None:
        self.http = HttpProbe()

    async def check_tei(self) -> dict[str, HealthCheckResult]:
        async def probe(url: str) -> HealthCheckResult:
            try:
                response, latency_ms = await self.http.get(f"{url.rstrip('/')}/health")
                response.raise_for_status()
                return HealthCheckResult(
                    status=HealthStatus.OK.value,
                    latency_ms=latency_ms,
                )
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("TEI health probe failed (%s)", type(exc).__name__)
                return HealthCheckResult(status=f"error: {exc}")

        embed, rerank = await asyncio.gather(probe(settings.tei_embed_url), probe(settings.tei_rerank_url))
        return {"tei_embed": embed, "tei_rerank": rerank}

    async def check_ollama(self) -> HealthCheckResult:
        try:
            response, latency_ms = await self.http.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
            return HealthCheckResult(status=HealthStatus.OK.value, latency_ms=latency_ms, models=models)
        except Exception as e:
            logger.warning("Ollama health probe failed (%s)", type(e).__name__)
            return HealthCheckResult(status=f"error: {e}")

    async def check_openrouter(self) -> HealthCheckResult:
        try:
            response, latency_ms = await self.http.get(
                f"{settings.openrouter_base_url}/models",
                request_timeout=5,
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            )
            if response.status_code < 400:
                return HealthCheckResult(
                    status=HealthStatus.OK.value,
                    latency_ms=latency_ms,
                    models=[settings.openrouter_model],
                )
            logger.warning("OpenRouter health probe returned HTTP %s", response.status_code)
            return HealthCheckResult(status=f"error: HTTP {response.status_code}")
        except Exception as e:
            logger.warning("Openrouter health probe failed (%s)", type(e).__name__)
            return HealthCheckResult(status=f"error: {e}")

    async def check_qdrant(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            await asyncio.to_thread(self.check_qdrant_sync)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return HealthCheckResult(status=HealthStatus.OK.value, latency_ms=latency_ms)
        except Exception as e:
            logger.warning("Qdrant health probe failed (%s)", type(e).__name__)
            return HealthCheckResult(status=f"error: {e}")

    def check_qdrant_sync(self) -> None:
        with closing(
            QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=3)
        ) as client:
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
            logger.warning("Postgres health probe failed (%s)", type(e).__name__)
            return HealthCheckResult(status=f"error: {e}")

    async def check_redis(self) -> HealthCheckResult:
        try:
            t0 = time.perf_counter()
            await redis_client.async_redis.ping()
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return HealthCheckResult(status=HealthStatus.OK.value, latency_ms=latency_ms)
        except Exception as e:
            logger.warning("Redis health probe failed (%s)", type(e).__name__)
            return HealthCheckResult(status=f"error: {e}")


class OllamaProbe:
    """Ollama model listing — delegates connectivity check to SystemHealthProbe."""

    def __init__(self, health_probe: SystemHealthProbe) -> None:
        self.health = health_probe

    async def get_models(self) -> list[str]:
        result = await self.health.check_ollama()
        return result.models or []


class QdrantInfo:
    """Provide collection details and reuse the system connectivity check."""

    def __init__(self, health_probe: SystemHealthProbe) -> None:
        self.health = health_probe

    async def get_status(self) -> str:
        return (await self.health.check_qdrant()).status

    async def get_collections(self) -> list[dict[str, Any]]:
        try:
            return await asyncio.to_thread(self.fetch_collections)
        except Exception as exc:
            logger.warning("Qdrant collection probe failed (%s)", type(exc).__name__)
            return []

    @staticmethod
    def fetch_collections() -> list[dict[str, Any]]:
        with closing(
            QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
        ) as client:
            return QdrantInfo.read_collections(client)

    @staticmethod
    def read_collections(client: QdrantClient) -> list[dict[str, Any]]:
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
