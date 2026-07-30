"""Pydantic models for request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    kind: str


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    kind: str
    is_active: bool


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "user"
    kind: str = "internal"


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    depth: str | None = None  # "short" | "detailed" | None (auto)


class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    sources: list[dict]


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class NewConversationResponse(BaseModel):
    conversation_id: int


class MessageResponse(BaseModel):
    role: str
    content: str
    sources: list[dict] | None = None


class ConversationHistoryResponse(BaseModel):
    conversation_id: int
    messages: list[MessageResponse]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    id: int
    filename: str
    visibility: str
    status: str
    error_message: str | None = None
    warning_message: str | None = None
    chunks: int | None = None
    chars: int | None = None
    owner_id: int | None = None


class UploadStatusResponse(BaseModel):
    status: str
    document_id: int
    filename: str


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    files: list[str]


class IngestStatusResponse(BaseModel):
    status: str
    mode: str | None = None
    file: str | None = None
    force: bool | None = None
    docs_dir: str | None = None


class IngestRegistryItem(BaseModel):
    filename: str
    chunks: int
    chars: int
    indexed_at: str
    source: str


class IngestRegistryResponse(BaseModel):
    total_files: int
    total_chunks: int
    files: list[IngestRegistryItem]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


class CreateGroupRequest(BaseModel):
    name: str


class GroupResponse(BaseModel):
    id: int
    name: str


class GroupMemberResponse(BaseModel):
    id: int
    email: str


class GroupMemberRequest(BaseModel):
    user_id: int


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class AssignClientRequest(BaseModel):
    internal_user_id: int


class ClientAssignmentResponse(BaseModel):
    internal_user_id: int
    email: str
    assigned_at: datetime


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    api: str
    qdrant: str
    ollama: str
    ollama_models: list[str] | None = None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    questions_path: str | None = None
    out_dir: str | None = None
    top_k: int | None = None
    judge_model: str | None = None


class BenchmarkResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Admin Config
# ---------------------------------------------------------------------------


class ConfigParamResponse(BaseModel):
    key: str
    value: str
    value_type: str
    category: str
    description: str | None = None
    min_value: float | None = None
    max_value: float | None = None


class ConfigParamUpdateRequest(BaseModel):
    value: str


class ModelsInfoResponse(BaseModel):
    llm_model: str
    embed_model: str
    rerank_model: str
    rerank_device: str
    ocr_engine: str
    ocr_enabled: bool
    ollama_models: list[str] | None = None


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


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


class StatusResponse(BaseModel):
    status: str
    detail: str | None = None
    id: int | None = None
    is_active: bool | None = None


class ApiKeyCreateRequest(BaseModel):
    name: str | None = None


class ApiKeyCreatedResponse(BaseModel):
    id: int
    api_key: str  # показывается только в этом ответе, больше нигде
    key_prefix: str
    name: str | None = None
    creation_date: datetime


class ApiKeyResponse(BaseModel):
    id: int
    key_prefix: str
    name: str | None = None
    creation_date: datetime
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None
    is_active: bool
