"""Application service for admin config endpoints (system info, models, vectordb)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from domain.value_objects.health_status import HealthStatus
from domain.value_objects.llm_provider import LLMProvider


@runtime_checkable
class OllamaProbePort(Protocol):
    async def get_models(self) -> list[str]: ...


@runtime_checkable
class VectorDBInfoPort(Protocol):
    def get_status(self) -> str: ...
    def get_collections(self) -> list[dict]: ...


@dataclass(frozen=True)
class ModelsInfo:
    llm_provider: str
    llm_model: str
    embed_model: str
    rerank_model: str
    device: str
    embed_device: str
    rerank_device: str
    ocr_engine: str
    ocr_enabled: bool
    ollama_models: list[str] = field(default_factory=list)
    openrouter_model: str | None = None


@dataclass(frozen=True)
class VectorDBCollectionInfo:
    name: str
    points_count: int = 0
    vectors_count: int = 0
    indexed_vectors_count: int = 0
    segments_count: int = 0
    status: str = ""
    optimizer_status: str = ""
    hnsw_m: int | None = None
    hnsw_ef_construct: int | None = None
    on_disk_payload: bool | None = None
    vector_size: int | None = None
    distance: str | None = None


@dataclass(frozen=True)
class VectorDBInfo:
    collections: list[VectorDBCollectionInfo] = field(default_factory=list)
    active_collection: str = ""
    qdrant_status: str = ""


@dataclass(frozen=True)
class OpenRouterModelsInfo:
    models: list[dict] = field(default_factory=list)
    active_model: str | None = None


class ConfigAdminService:
    def __init__(
        self,
        ollama_probe: OllamaProbePort,
        vectordb_info: VectorDBInfoPort,
        openrouter_models_fetcher=None,
        *,
        llm_provider: str = "",
        llm_model: str = "",
        embed_model: str = "",
        rerank_model: str = "",
        device: str = "",
        embed_device: str = "",
        rerank_device: str = "",
        ocr_engine: str = "",
        ocr_enabled: bool = False,
        openrouter_model: str | None = None,
        active_collection: str = "",
    ) -> None:
        self._ollama = ollama_probe
        self._vectordb = vectordb_info
        self._openrouter_fetcher = openrouter_models_fetcher
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._embed_model = embed_model
        self._rerank_model = rerank_model
        self._device = device
        self._embed_device = embed_device
        self._rerank_device = rerank_device
        self._ocr_engine = ocr_engine
        self._ocr_enabled = ocr_enabled
        self._openrouter_model = openrouter_model
        self._active_collection = active_collection

    async def get_models_info(self) -> ModelsInfo:
        ollama_models = await self._ollama.get_models()
        return ModelsInfo(
            llm_provider=self._llm_provider,
            llm_model=self._llm_model,
            embed_model=self._embed_model,
            rerank_model=self._rerank_model,
            device=self._device,
            embed_device=self._embed_device,
            rerank_device=self._rerank_device,
            ocr_engine=self._ocr_engine,
            ocr_enabled=self._ocr_enabled,
            ollama_models=ollama_models,
            openrouter_model=self._openrouter_model
            if self._llm_provider == LLMProvider.OPENROUTER.value
            else None,
        )

    def get_vectordb_info(self) -> VectorDBInfo:
        qdrant_status = self._vectordb.get_status()
        collections_raw = self._vectordb.get_collections() if qdrant_status == HealthStatus.OK.value else []

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
            active_collection=self._active_collection,
            qdrant_status=qdrant_status,
        )

    async def get_openrouter_models(self) -> OpenRouterModelsInfo:
        if self._openrouter_fetcher is None:
            return OpenRouterModelsInfo(models=[], active_model=None)
        models = await self._openrouter_fetcher()
        return OpenRouterModelsInfo(models=models, active_model=self._openrouter_model)
