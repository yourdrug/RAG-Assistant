"""Admin config & models info schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigParamResponse(BaseModel):
    key: str
    value: str
    value_type: str
    category: str | None = None
    description: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    domain_key: str | None = None


class ConfigParamUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class ModelsInfoResponse(BaseModel):
    llm_provider: str
    llm_model: str
    tei_embed_url: str
    tei_rerank_url: str
    embed_model: str
    rerank_model: str
    ocr_engine: str
    ocr_enabled: bool
    ollama_models: list[str]
    openrouter_model: str | None = None
    ml_provider: str = "tei"


class OpenRouterModelInfo(BaseModel):
    id: str
    name: str
    context_length: int
    pricing: dict[str, float]


class OpenRouterModelsResponse(BaseModel):
    models: list[OpenRouterModelInfo]
    active_model: str


class VectorDBCollectionInfo(BaseModel):
    name: str
    points_count: int
    vectors_count: int
    indexed_vectors_count: int
    segments_count: int
    status: str
    optimizer_status: str
    hnsw_m: int | None = None
    hnsw_ef_construct: int | None = None
    on_disk_payload: bool | None = None
    vector_size: int | None = None
    distance: str | None = None


class VectorDBInfoResponse(BaseModel):
    collections: list[VectorDBCollectionInfo]
    active_collection: str
    qdrant_status: str
