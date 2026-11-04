"""Pydantic schemas for request / response validation.

Re-exports all classes so that ``from presentation.api.schemas import X``
continues to work after the split into per-domain modules.
"""

from presentation.api.schemas.admin_act_versions import (
    ActSummary,
    ActVersionListResponse,
    ActVersionReviewItem,
    ActVersionUpdateRequest,
    ActVersionUpdateResponse,
)
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
    IndexFromPreviewResponse,
    PageDiagnostic,
    PageImageResponse,
    PreviewFile,
)
from presentation.api.schemas.api_keys import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
    ApiKeyRevokeResponse,
)
from presentation.api.schemas.auth import (
    ChangeRoleRequest,
    CreateUserRequest,
    CuratorScopeResponse,
    CurrentUser,
    LoginRequest,
    MeResponse,
    TokenResponse,
    UserListResponse,
    UserResponse,
    UserToggleResponse,
)
from presentation.api.schemas.benchmark import (
    BenchmarkCancelResponse,
    BenchmarkDeleteResponse,
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
    SourceFilesResponse,
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
    ChunkDeleteResponse,
    ChunkEditRequest,
    ChunkListResponse,
    ChunkResponse,
)
from presentation.api.schemas.curators import CuratorAssignmentResponse
from presentation.api.schemas.documents import (
    DeleteDocumentResponse,
    DocumentRenameRequest,
    DocumentResponse,
    ManualDocumentRequest,
    UploadResponse,
    UploadStatusResponse,
)
from presentation.api.schemas.groups import (
    CreateGroupRequest,
    GroupAssignResponse,
    GroupMemberRequest,
    GroupMemberResponse,
    GroupResponse,
)
from presentation.api.schemas.health import HealthCheck, HealthResponse
from presentation.api.schemas.ingest import IngestRegistryItem, IngestRegistryResponse, IngestStatusResponse
from presentation.api.schemas.search import ExactSearchRequest, ExactSearchResponse, ExactSearchResult
from presentation.api.schemas.validators import FileContent, SafeRelativePath


__all__ = [
    # admin_act_versions
    "ActSummary",
    "ActVersionListResponse",
    "ActVersionReviewItem",
    "ActVersionUpdateRequest",
    "ActVersionUpdateResponse",
    # api_keys
    "ApiKeyCreateRequest",
    "ApiKeyCreateResponse",
    "ApiKeyResponse",
    "ApiKeyRevokeResponse",
    # benchmark
    "BenchmarkCancelResponse",
    "BenchmarkDeleteResponse",
    "BenchmarkHistoryPoint",
    "BenchmarkHistoryResponse",
    "BenchmarkQuestionCreate",
    "BenchmarkQuestionResponse",
    "BenchmarkQuestionUpdate",
    "BenchmarkQuestionsImportRequest",
    "BenchmarkQuestionsImportResponse",
    "BenchmarkQuestionsListResponse",
    "BenchmarkRequest",
    "BenchmarkResponse",
    "BenchmarkResultDetail",
    "BenchmarkResultSummary",
    "BenchmarkResultsListResponse",
    "BenchmarkRunResponse",
    "BenchmarkRunsListResponse",
    # auth
    "ChangeRoleRequest",
    # admin_logs
    "ChatLogEntry",
    "ChatLogsResponse",
    # chat
    "ChatRequest",
    "ChatResponse",
    # chunks
    "ChunkCreateRequest",
    "ChunkCursorListResponse",
    "ChunkDeleteResponse",
    "ChunkEditRequest",
    "ChunkListResponse",
    "ChunkResponse",
    # admin_config
    "ConfigParamResponse",
    "ConfigParamUpdateRequest",
    "ConversationHistoryResponse",
    "ConversationListItem",
    "ConversationListResponse",
    # groups
    "CreateGroupRequest",
    "CreateUserRequest",
    # curators
    "CuratorAssignmentResponse",
    "CuratorScopeResponse",
    "CurrentUser",
    # documents
    "DeleteDocumentResponse",
    # admin_quality
    "DocumentDiagnoseResponse",
    "DocumentQualityItem",
    "DocumentQualityListResponse",
    "DocumentRenameRequest",
    "DocumentResponse",
    "DryRunPageResult",
    "DryRunResponse",
    # search
    "ExactSearchRequest",
    "ExactSearchResponse",
    "ExactSearchResult",
    # validators
    "FileContent",
    "GroupAssignResponse",
    "GroupMemberRequest",
    "GroupMemberResponse",
    "GroupResponse",
    # health
    "HealthCheck",
    "HealthResponse",
    "IndexFromPreviewResponse",
    # ingest
    "IngestRegistryItem",
    "IngestRegistryResponse",
    "IngestStatusResponse",
    # admin_jobs
    "JobResponse",
    "JobsListResponse",
    "JobsStatsResponse",
    "LogEntry",
    "LoginRequest",
    "LogsResponse",
    "ManualDocumentRequest",
    "MeResponse",
    "MessageResponse",
    # admin_metrics
    "MetricsResponse",
    "ModelsInfoResponse",
    "NewConversationResponse",
    "OpenRouterModelInfo",
    "OpenRouterModelsResponse",
    "PageDiagnostic",
    "PageImageResponse",
    "PreviewFile",
    "RegressionCheckResponse",
    "RegressionCheckResult",
    "RunApplyFailed",
    "RunApplyResponse",
    "RunCompareResponse",
    "SafeRelativePath",
    "SourceFilesResponse",
    "SweepCreateRequest",
    "SweepResponse",
    "SweepsListResponse",
    "TokenResponse",
    "UploadResponse",
    "UploadStatusResponse",
    "UserListResponse",
    "UserResponse",
    "UserToggleResponse",
    "VectorDBCollectionInfo",
    "VectorDBInfoResponse",
]
