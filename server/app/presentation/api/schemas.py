"""Pydantic schemas for request / response validation."""

from __future__ import annotations

from datetime import datetime

from domain.value_objects.benchmark_strategy import BenchmarkStrategy
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.search_mode import SearchMode
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
    llm_provider: str
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


class BenchmarkResultSummary(BaseModel):
    id: int
    config_json: dict
    summary_metrics: dict
    duration_sec: float
    llm_evaluated: bool
    dataset: str
    sweep_id: int | None = None
    creation_date: datetime | None = None


class BenchmarkResultDetail(BaseModel):
    id: int
    summary: BenchmarkResultSummary
    per_question_results: dict | None = None


class BenchmarkResultsListResponse(BaseModel):
    results: list[BenchmarkResultSummary]
    total: int


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
    quality_score: float | None = None
    chunks: int | None
    chars: int | None
    creation_date: datetime | None
    indexed_at: datetime | None
    source_type: str = "file"
    has_manual_edits: bool = False


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
    confidence: float | None = Field(None, description="Answer confidence score (0.0-1.0)")


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


class GroupCreateRequest(BaseModel):
    name: str


class GroupResponse(BaseModel):
    id: int
    name: str
    creation_date: datetime | None = None


class GroupMemberResponse(BaseModel):
    model_config = {"populate_by_name": True}
    id: int = Field(validation_alias="user_id")
    email: str


class AddMemberRequest(BaseModel):
    user_id: int


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
    llm_provider: str
    llm_model: str
    embed_model: str
    rerank_model: str
    device: str
    embed_device: str
    rerank_device: str
    ocr_engine: str
    ocr_enabled: bool
    ollama_models: list[str]
    openrouter_model: str | None = None


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
# Chat Logs (Q&A quality tracking)
# ---------------------------------------------------------------------------


class ChatLogEntry(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    creation_date: str
    user_id: int | None = None
    conversation_id: int | None = None
    question: str
    answer: str
    sources: list | None = None
    latency_ms: int | None = None
    model_used: str | None = None
    breadth: str | None = None
    domain: str | None = None
    retrieval_count: int | None = None
    reranker_score: float | None = None


class ChatLogsResponse(BaseModel):
    logs: list[ChatLogEntry]
    total: int


# ---------------------------------------------------------------------------
# Exact Substring Search (pg_trgm)
# ---------------------------------------------------------------------------


class ExactSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=200, description="Search query (min 3 chars)")
    mode: str = Field(
        SearchMode.EXACT.value,
        pattern="^(exact|icontains)$",
        description="exact=pg_trgm ranked, icontains=plain ILIKE",
    )
    limit: int = Field(20, ge=1, le=100)
    document_id: int | None = Field(None, description="Filter by document ID (optional)")


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


# ---------------------------------------------------------------------------
# Chunk Management
# ---------------------------------------------------------------------------


class ChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    filename: str = ""
    visibility: str = ""
    doc_domain: str = DocDomain.GENERAL.value
    owner_id: int | None = None
    group_id: int | None = None
    edited_at: str | None = None
    edited_by: int | None = None
    manual: bool = False
    creation_date: str | None = None
    warning: str | None = None


class ChunkCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    page: int | None = Field(None, description="Page number (optional)")
    section: str | None = Field(None, description="Section name (optional)")


class ChunkEditRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class ChunkListResponse(BaseModel):
    chunks: list[ChunkResponse]
    total: int
    document_id: int


class ManualDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    visibility: str
    group_id: int | None = None


# ---------------------------------------------------------------------------
# Benchmark Lab — Questions
# ---------------------------------------------------------------------------


class BenchmarkQuestionCreate(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    expected_answer: str | None = None
    source_hint: str | None = None
    tags: list[str] | None = None
    dataset: str = "main"
    notes: str | None = None


class BenchmarkQuestionUpdate(BaseModel):
    question: str | None = None
    expected_answer: str | None = None
    source_hint: str | None = None
    tags: list[str] | None = None
    dataset: str | None = None
    is_active: bool | None = None
    notes: str | None = None


class BenchmarkQuestionResponse(BaseModel):
    id: int
    question: str
    expected_answer: str | None = None
    source_hint: str | None = None
    tags: list[str] | None = None
    dataset: str
    is_active: bool
    created_by: int | None = None
    notes: str | None = None
    creation_date: datetime | None = None


class BenchmarkQuestionsListResponse(BaseModel):
    questions: list[BenchmarkQuestionResponse]
    total: int


class BenchmarkQuestionsImportRequest(BaseModel):
    questions: list[BenchmarkQuestionCreate]


class BenchmarkQuestionsImportResponse(BaseModel):
    imported: int


# ---------------------------------------------------------------------------
# Benchmark Lab — Sweeps
# ---------------------------------------------------------------------------


class SweepCreateRequest(BaseModel):
    strategy: str = Field(BenchmarkStrategy.GRID.value, pattern="^(grid|random|successive_halving)$")
    search_space: dict
    objective_weights: dict = Field(
        default_factory=lambda: {"hit_rate": 0.4, "faithfulness": 0.3, "relevancy": 0.3}
    )
    dataset: str = "main"
    top_n_llm: int = Field(3, ge=0, le=20)


class SweepResponse(BaseModel):
    id: int
    status: str
    strategy: str
    search_space: dict
    objective_weights: dict
    dataset: str
    top_n_llm: int
    total_configs: int
    evaluated_configs: int
    best_run_id: int | None = None
    job_id: int | None = None
    creation_date: datetime | None = None


class SweepsListResponse(BaseModel):
    sweeps: list[SweepResponse]
    total: int


# ---------------------------------------------------------------------------
# Benchmark Lab — Runs
# ---------------------------------------------------------------------------


class BenchmarkRunResponse(BaseModel):
    id: int
    sweep_id: int | None = None
    config_json: dict
    summary_metrics: dict
    duration_sec: float
    llm_evaluated: bool
    dataset: str
    filename: str | None = None
    creation_date: datetime | None = None


class BenchmarkRunsListResponse(BaseModel):
    runs: list[BenchmarkRunResponse]
    total: int


class RunApplyResponse(BaseModel):
    applied: int
    keys: list[str]


class RunCompareResponse(BaseModel):
    runs: list[BenchmarkRunResponse]
    diff: dict


# ---------------------------------------------------------------------------
# Benchmark Lab — History / Trends
# ---------------------------------------------------------------------------


class BenchmarkHistoryPoint(BaseModel):
    run_id: int
    creation_date: datetime | None = None
    metrics: dict
    config_summary: dict
    dataset: str
    llm_evaluated: bool


class BenchmarkHistoryResponse(BaseModel):
    points: list[BenchmarkHistoryPoint]
    total: int


# ---------------------------------------------------------------------------
# Admin Quality / Diagnostics
# ---------------------------------------------------------------------------


class DocumentQualityItem(BaseModel):
    id: int
    filename: str
    status: str
    quality_score: float | None = None
    warning_message: str | None = None
    chunks: int | None = None
    chars: int | None = None
    indexed_at: datetime | None = None


class DocumentQualityListResponse(BaseModel):
    documents: list[DocumentQualityItem]
    total: int


class PageDiagnostic(BaseModel):
    page: int
    type: str  # text, scan, garbled, empty, table
    chars: int
    description: str


class DocumentDiagnoseResponse(BaseModel):
    document_id: int
    filename: str
    total_pages: int
    pages: list[PageDiagnostic]
    summary: dict


class DryRunPageResult(BaseModel):
    page: int
    type: str  # text, scan, garbled, empty, table
    content_type: str  # text, table, ocr
    chars: int
    preview: str  # first 200 chars


class DryRunResponse(BaseModel):
    filename: str
    total_pages: int
    pages: list[DryRunPageResult]
    total_chars: int
    quality_score: float
    warning: str | None = None
    full_text_preview: str  # first 2000 chars
    summary: dict  # {text: N, scan: N, garbled: N, empty: N, table: N}
