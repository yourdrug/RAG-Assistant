"""Pydantic schemas for request / response validation."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthCheck(BaseModel):
    status: str
    latency_ms: float | None = None
    models: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    checks: dict[str, HealthCheck]
    background_jobs: dict[str, int]


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
# Documents
# ---------------------------------------------------------------------------


class UploadStatusResponse(BaseModel):
    status: str
    document_id: int
    filename: str


class DocumentResponse(BaseModel):
    id: int
    filename: str
    source_path: str
    visibility: str
    owner_id: int | None
    group_id: int | None
    status: str
    error_message: str | None
    warning_message: str | None
    chunks: int | None
    chars: int | None
    creation_date: datetime | None
    indexed_at: datetime | None


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestStatusResponse(BaseModel):
    status: str
    mode: str | None = None
    docs_dir: str | None = None
    file: str | None = None
    force: bool | None = None


class IngestRegistryItem(BaseModel):
    filename: str
    chunks: int | None = None
    chars: int | None = None
    indexed_at: datetime | None = None
    source: str | None = None


class IngestRegistryResponse(BaseModel):
    total_files: int
    total_chunks: int
    files: list[IngestRegistryItem]


class UploadResponse(BaseModel):
    files: list[str]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class ConversationCreateRequest(BaseModel):
    pass


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    creation_date: datetime | None


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: list | None
    creation_date: datetime | None


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    depth: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    sources: list[dict] | None = None


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


class GroupCreateRequest(BaseModel):
    name: str


class GroupResponse(BaseModel):
    id: int
    name: str
    creation_date: datetime | None


class GroupMemberResponse(BaseModel):
    model_config = {"populate_by_name": True}
    id: int = Field(validation_alias="user_id")
    email: str


class AddMemberRequest(BaseModel):
    user_id: int


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class ClientAssignmentResponse(BaseModel):
    internal_user_id: int
    email: str
    assigned_at: datetime | None


class AssignClientRequest(BaseModel):
    client_user_id: int


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------


class ApiKeyCreateRequest(BaseModel):
    name: str


class ApiKeyCreateResponse(BaseModel):
    key: str
    key_prefix: str
    name: str | None


class ApiKeyResponse(BaseModel):
    id: int
    key_prefix: str
    name: str | None
    created_at: datetime | None
    last_used_at: datetime | None
    revoked_at: datetime | None
    is_active: bool


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str | None = None
    kind: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    kind: str
    is_active: bool


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class NewConversationResponse(BaseModel):
    conversation_id: int


class ConversationHistoryResponse(BaseModel):
    conversation_id: int
    messages: list[MessageResponse]


class ConversationListItem(BaseModel):
    id: int
    title: str | None = None
    created_at: datetime | None = None
    message_count: int = 0


class ConversationListResponse(BaseModel):
    conversations: list[ConversationListItem]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


class CreateGroupRequest(BaseModel):
    name: str


class GroupMemberRequest(BaseModel):
    user_id: int


# ---------------------------------------------------------------------------
# Admin Config
# ---------------------------------------------------------------------------


class ConfigParamResponse(BaseModel):
    key: str
    value: str
    value_type: str
    category: str | None = None
    description: str | None = None
    min_value: float | None = None
    max_value: float | None = None


class ConfigParamUpdateRequest(BaseModel):
    value: str


class ModelsInfoResponse(BaseModel):
    llm_model: str
    embed_model: str
    rerank_model: str
    device: str
    embed_device: str
    rerank_device: str
    ocr_engine: str
    ocr_enabled: bool
    ollama_models: list[str]


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
# Background Jobs
# ---------------------------------------------------------------------------


class JobResponse(BaseModel):
    id: int
    job_type: str
    status: str
    related_id: int | None = None
    request_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    creation_date: datetime | None = None


class JobsListResponse(BaseModel):
    total: int
    jobs: list[JobResponse]


class JobsStatsResponse(BaseModel):
    total: int
    by_status: dict[str, int]


# ---------------------------------------------------------------------------
# Monitoring / Metrics
# ---------------------------------------------------------------------------


class MetricValue(BaseModel):
    name: str
    value: float
    labels: dict[str, str] = {}


class MetricsResponse(BaseModel):
    db_pool: dict[str, float]
    qdrant: dict[str, float]
    bm25: dict[str, float]
    ollama: list[dict[str, object]]
    rag: dict[str, object]
    ingestion: dict[str, object]
    http_requests: dict[str, object]


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


class LogEntry(BaseModel):
    timestamp: str
    level: str
    logger: str
    request_id: str
    message: str
    filename: str | None = None
    lineno: int | None = None


class LogsResponse(BaseModel):
    logs: list[LogEntry]
    total: int


# ---------------------------------------------------------------------------
# Exact Substring Search (pg_trgm)
# ---------------------------------------------------------------------------


class ExactSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=200, description="Search query (min 3 chars)")
    mode: str = Field(
        "exact", pattern="^(exact|icontains)$", description="exact=pg_trgm ranked, icontains=plain ILIKE"
    )
    limit: int = Field(20, ge=1, le=100)


class ExactSearchResult(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    content: str
    chunk_index: int


class ExactSearchResponse(BaseModel):
    query: str
    results: list[ExactSearchResult]
    total: int
