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
    llm_base_url: str
    llm_model: str
    llm_auxiliary_model: str
    llm_api_key_configured: bool
    embedding_base_url: str
    embed_model: str
    embedding_api_key_configured: bool
    rerank_base_url: str
    rerank_model: str
    rerank_api_format: str
    rerank_api_key_configured: bool
    ocr_enabled: bool


class LLMModelInfo(BaseModel):
    id: str
    name: str
    context_length: int | None = None
    # Present only when the endpoint reports it (e.g. OpenRouter); USD per 1M tokens.
    pricing: dict[str, float] | None = None


class LLMModelsResponse(BaseModel):
    models: list[LLMModelInfo]
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
