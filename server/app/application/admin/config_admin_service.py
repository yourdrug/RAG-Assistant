"""Application service for admin system information."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from application.admin.dto import ModelsInfo, OpenRouterModelsInfo, VectorDBCollectionInfo, VectorDBInfo
from domain.value_objects.health_status import HealthStatus
from domain.value_objects.llm_provider import LLMProvider
from domain.value_objects.ml_provider import MLProvider


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.admin.ports import OllamaProbePort, VectorDBInfoPort
    from application.admin.ports.settings import ConfigAdminSettingsPort


class ConfigAdminService:
    def __init__(
        self,
        ollama_probe: OllamaProbePort,
        vectordb_info: VectorDBInfoPort,
        admin_settings: ConfigAdminSettingsPort,
        openrouter_models_fetcher: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
    ) -> None:
        self._ollama = ollama_probe
        self._vectordb = vectordb_info
        self._settings = admin_settings
        self._openrouter_fetcher = openrouter_models_fetcher

    async def get_models_info(self) -> ModelsInfo:
        ollama_models = await self._ollama.get_models()

        if self._settings.ml_provider == MLProvider.DEEPINFRA.value:
            embed_model = self._settings.deepinfra_embed_model
            rerank_model = self._settings.deepinfra_rerank_model
        else:
            embed_model = self._settings.tei_embed_url
            rerank_model = self._settings.tei_rerank_url

        return ModelsInfo(
            llm_provider=self._settings.llm_provider,
            llm_model=self._settings.llm_model,
            tei_embed_url=self._settings.tei_embed_url,
            tei_rerank_url=self._settings.tei_rerank_url,
            embed_model=embed_model,
            rerank_model=rerank_model,
            ocr_engine=self._settings.ocr_engine,
            ocr_enabled=self._settings.ocr_enabled,
            ollama_models=ollama_models,
            openrouter_model=self._settings.openrouter_model
            if self._settings.llm_provider == LLMProvider.OPENROUTER.value
            else None,
            ml_provider=self._settings.ml_provider,
        )

    async def get_vectordb_info(self) -> VectorDBInfo:
        qdrant_status = await self._vectordb.get_status()
        collections_raw = (
            await self._vectordb.get_collections() if qdrant_status == HealthStatus.OK.value else []
        )

        collections = [
            VectorDBCollectionInfo(
                name=c["name"],
                points_count=c.get("points_count", 0),
                vectors_count=c.get("vectors_count", 0),
                indexed_vectors_count=c.get("indexed_vectors_count", 0),
                segments_count=c.get("segments_count", 0),
                status=c.get("status", ""),
                optimizer_status=c.get("optimizer_status", ""),
                hnsw_m=c.get("hnsw_m"),
                hnsw_ef_construct=c.get("hnsw_ef_construct"),
                on_disk_payload=c.get("on_disk_payload"),
                vector_size=c.get("vector_size"),
                distance=c.get("distance"),
            )
            for c in collections_raw
        ]

        return VectorDBInfo(
            collections=collections,
            active_collection=self._settings.collection_name,
            qdrant_status=qdrant_status,
        )

    async def get_openrouter_models(self) -> OpenRouterModelsInfo:
        if self._openrouter_fetcher is None:
            return OpenRouterModelsInfo(models=[], active_model=None)

        models = await self._openrouter_fetcher()
        return OpenRouterModelsInfo(models=models, active_model=self._settings.openrouter_model)
