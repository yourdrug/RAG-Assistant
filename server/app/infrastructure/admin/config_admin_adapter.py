"""Adapters for ConfigAdminService infrastructure ports."""

from __future__ import annotations

import httpx
from config import settings
from domain.value_objects.health_status import HealthStatus
from qdrant_client import QdrantClient


class OllamaProbe:
    async def get_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{settings.ollama_base_url}/api/tags")
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []


class QdrantInfo:
    def get_status(self) -> str:
        try:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
            client.get_collections()
            return HealthStatus.OK.value
        except Exception as e:
            return f"error: {e}"

    def get_collections(self) -> list[dict]:
        try:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
            col_list = client.get_collections()
            result = []
            for col in col_list.collections:
                info = client.get_collection(col.name)
                vectors_cfg = info.config.params.vectors if info.config.params.vectors else None
                hnsw_cfg = info.config.hnsw_config
                result.append(
                    {
                        "name": col.name,
                        "points_count": info.points_count or 0,
                        "vectors_count": info.vectors_count or 0,
                        "indexed_vectors_count": info.indexed_vectors_count or 0,
                        "segments_count": info.segments_count or 0,
                        "status": str(info.status.value)
                        if hasattr(info.status, "value")
                        else str(info.status),
                        "optimizer_status": (
                            str(info.optimizer_status.value)
                            if hasattr(info.optimizer_status, "value")
                            else str(info.optimizer_status)
                        ),
                        "hnsw_m": hnsw_cfg.m if hnsw_cfg else None,
                        "hnsw_ef_construct": hnsw_cfg.ef_construct if hnsw_cfg else None,
                        "on_disk_payload": info.config.params.on_disk_payload if info.config.params else None,
                        "vector_size": vectors_cfg.size if vectors_cfg else None,
                        "distance": (
                            str(vectors_cfg.distance.value)
                            if vectors_cfg and hasattr(vectors_cfg.distance, "value")
                            else str(vectors_cfg.distance)
                            if vectors_cfg
                            else None
                        ),
                    }
                )
            return result
        except Exception:
            return []


async def fetch_openrouter_models() -> list[dict]:
    from infrastructure.ml.factories import fetch_openrouter_models as _fetch

    return await _fetch()
