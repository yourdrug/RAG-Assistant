"""Pydantic schemas for request / response validation.

Re-exports all classes so that ``from presentation.api.schemas import X``
continues to work after the split into per-domain modules.
"""

from presentation.api.schemas.admin_config import (
    ConfigParamResponse,
    ConfigParamUpdateRequest,
    ModelsInfoResponse,
    OpenRouterModelInfo,
    OpenRouterModelsResponse,
    VectorDBCollectionInfo,
    VectorDBInfoResponse,
)
from presentation.api.schemas.admin_jobs import JobResponse, JobsListResponse, JobsStatsResponse
from presentation.api.schemas.admin_logs import (
    ChatLogEntry,
    ChatLogsResponse,
    LogEntry,
    LogsResponse,
)
from presentation.api.schemas.admin_metrics import MetricsResponse
from presentation.api.schemas.admin_quality import (
    DocumentDiagnoseResponse,
    DocumentQualityItem,
    DocumentQualityListResponse,
    DryRunPageResult,
    DryRunResponse,
    PageDiagnostic,
    PageImageResponse,
)
from presentation.api.schemas.api_keys import ApiKeyCreateRequest, ApiKeyCreateResponse, ApiKeyResponse
from presentation.api.schemas.auth import (
    ChangeRoleRequest,
    CreateUserRequest,
    CuratorScopeResponse,
    LoginRequest,
    TokenResponse,
    UserListResponse,
    UserResponse,
)
from presentation.api.schemas.benchmark import (
    BenchmarkHistoryPoint,
    BenchmarkHistoryResponse,
    BenchmarkQuestionCreate,
    BenchmarkQuestionResponse,
    BenchmarkQuestionsImportRequest,
    BenchmarkQuestionsImportResponse,
    BenchmarkQuestionsListResponse,
    BenchmarkQuestionUpdate,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResultDetail,
    BenchmarkResultsListResponse,
    BenchmarkResultSummary,
    BenchmarkRunResponse,
    BenchmarkRunsListResponse,
    RegressionCheckResponse,
    RegressionCheckResult,
    RunApplyFailed,
    RunApplyResponse,
    RunCompareResponse,
    SweepCreateRequest,
    SweepResponse,
    SweepsListResponse,
)
from presentation.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    ConversationListItem,
    ConversationListResponse,
    MessageResponse,
    NewConversationResponse,
)
from presentation.api.schemas.chunks import (
    ChunkCreateRequest,
    ChunkCursorListResponse,
    ChunkEditRequest,
    ChunkListResponse,
    ChunkResponse,
)
from presentation.api.schemas.documents import (
    DocumentRenameRequest,
    DocumentResponse,
    ManualDocumentRequest,
    UploadResponse,
    UploadStatusResponse,
)
from presentation.api.schemas.groups import (
    CreateGroupRequest,
    GroupMemberRequest,
    GroupMemberResponse,
    GroupResponse,
)
from presentation.api.schemas.health import HealthCheck, HealthResponse
from presentation.api.schemas.ingest import IngestRegistryItem, IngestRegistryResponse, IngestStatusResponse
from presentation.api.schemas.search import ExactSearchRequest, ExactSearchResponse, ExactSearchResult

__all__ = [
    # admin_config
    "ConfigParamResponse",
    "ConfigParamUpdateRequest",
    "ModelsInfoResponse",
    "OpenRouterModelInfo",
    "OpenRouterModelsResponse",
    "VectorDBCollectionInfo",
    "VectorDBInfoResponse",
    # admin_jobs
    "JobResponse",
    "JobsListResponse",
    "JobsStatsResponse",
    # admin_logs
    "ChatLogEntry",
    "ChatLogsResponse",
    "LogEntry",
    "LogsResponse",
    # admin_metrics
    "MetricsResponse",
    # admin_quality
    "DocumentDiagnoseResponse",
    "DocumentQualityItem",
    "DocumentQualityListResponse",
    "DryRunPageResult",
    "DryRunResponse",
    "PageDiagnostic",
    "PageImageResponse",
    # api_keys
    "ApiKeyCreateRequest",
    "ApiKeyCreateResponse",
    "ApiKeyResponse",
    # auth
    "ChangeRoleRequest",
    "CreateUserRequest",
    "CuratorScopeResponse",
    "LoginRequest",
    "TokenResponse",
    "UserListResponse",
    "UserResponse",
    # benchmark
    "BenchmarkHistoryPoint",
    "BenchmarkHistoryResponse",
    "BenchmarkQuestionCreate",
    "BenchmarkQuestionResponse",
    "BenchmarkQuestionsImportRequest",
    "BenchmarkQuestionsImportResponse",
    "BenchmarkQuestionsListResponse",
    "BenchmarkQuestionUpdate",
    "BenchmarkRequest",
    "BenchmarkResponse",
    "BenchmarkResultDetail",
    "BenchmarkResultsListResponse",
    "BenchmarkResultSummary",
    "BenchmarkRunResponse",
    "BenchmarkRunsListResponse",
    "RegressionCheckResponse",
    "RegressionCheckResult",
    "RunApplyFailed",
    "RunApplyResponse",
    "RunCompareResponse",
    "SweepCreateRequest",
    "SweepResponse",
    "SweepsListResponse",
    # chat
    "ChatRequest",
    "ChatResponse",
    "ConversationHistoryResponse",
    "ConversationListItem",
    "ConversationListResponse",
    "MessageResponse",
    "NewConversationResponse",
    # chunks
    "ChunkCreateRequest",
    "ChunkCursorListResponse",
    "ChunkEditRequest",
    "ChunkListResponse",
    "ChunkResponse",
    # documents
    "DocumentRenameRequest",
    "DocumentResponse",
    "ManualDocumentRequest",
    "UploadResponse",
    "UploadStatusResponse",
    # groups
    "CreateGroupRequest",
    "GroupMemberRequest",
    "GroupMemberResponse",
    "GroupResponse",
    # health
    "HealthCheck",
    "HealthResponse",
    # ingest
    "IngestRegistryItem",
    "IngestRegistryResponse",
    "IngestStatusResponse",
    # search
    "ExactSearchRequest",
    "ExactSearchResponse",
    "ExactSearchResult",
]
