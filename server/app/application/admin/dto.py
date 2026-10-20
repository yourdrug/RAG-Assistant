"""Public configuration and system information models."""

from dataclasses import dataclass, field
from typing import Any

from domain.runtime.enums import MLProvider


@dataclass(frozen=True)
class ModelsInfo:
    llm_provider: str
    llm_model: str
    tei_embed_url: str
    tei_rerank_url: str
    embed_model: str
    rerank_model: str
    ocr_engine: str
    ocr_enabled: bool
    ollama_models: list[str] = field(default_factory=list)
    openrouter_model: str | None = None
    ml_provider: str = MLProvider.TEI.value


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
    models: list[dict[str, Any]] = field(default_factory=list)
    active_model: str | None = None


@dataclass(frozen=True)
class ConfigurationStatus:
    saved_version: str | None
    applied_version: str | None
    ready: bool
    error: str | None


@dataclass(frozen=True)
class EffectiveParameter:
    key: str
    source: str
    configured_value: str | None
    applied_value: str | None
    editable: bool
    application_mode: str
    present: bool | None = None
    domain_key: str | None = None
