"""Read-only view of assembled dependencies consumed by HTTP providers."""

from __future__ import annotations

from typing import Protocol

from application.admin import (
    ConfigAdminService,
    ConfigMaskerPort,
    ConfigService,
    CorpusControlPort,
    LogBufferPort,
)
from application.admin.metrics import MetricsService
from application.benchmark import (
    BenchmarkExportService,
    BenchmarkHistoryPort,
    BenchmarkQuestionService,
    BenchmarkResultService,
    BenchmarkRunService,
    BenchmarkSweepLauncher,
    BenchmarkSweepService,
)
from application.budget import BudgetAdminService
from application.chat import ChatLogService, ChatService, ConversationService
from application.chunks import ChunkMutationService, ChunkQueryService
from application.documents import (
    DocumentCommandService,
    DocumentDeletionJobs,
    DocumentQueryService,
    ManualDocumentService,
    PreviewCache,
    PreviewStrategyFactoryPort,
)
from application.documents.diagnostics import PDFDiagnosticService
from application.documents.quality_service import QualityService
from application.health.service import HealthService
from application.identity import (
    ApiKeyProviderPort,
    ApiKeyService,
    AssignmentService,
    AuthService,
    GroupService,
    UserManagementService,
)
from application.ingestion import IngestionService
from application.jobs.service import JobService
from application.ports.action_logger import ActionLoggerPort
from application.ports.cache_invalidator import CacheInvalidatorPort
from application.ports.idempotency_store import IdempotencyStorePort
from application.ports.job_enqueuer import JobEnqueuerPort
from application.ports.rate_limit import RateLimiterPort
from application.search import SearchService
from application.uow.factory import UnitOfWorkFactory
from domain.ingestion.ports import DomainSettingsPort
from domain.ingestion.profiles.registry import DomainProfileRegistry


class ApplicationServices(Protocol):
    @property
    def chat_service(self) -> ChatService | None: ...

    @property
    def auth_service(self) -> AuthService | None: ...

    @property
    def user_management_service(self) -> UserManagementService | None: ...

    @property
    def api_key_service(self) -> ApiKeyService | None: ...

    @property
    def document_command_service(self) -> DocumentCommandService | None: ...

    @property
    def document_query_service(self) -> DocumentQueryService | None: ...

    @property
    def document_deletion_jobs(self) -> DocumentDeletionJobs | None: ...

    @property
    def chunk_query_service(self) -> ChunkQueryService | None: ...

    @property
    def chunk_mutation_service(self) -> ChunkMutationService | None: ...

    @property
    def manual_document_service(self) -> ManualDocumentService | None: ...

    @property
    def config_service(self) -> ConfigService | None: ...

    @property
    def health_service(self) -> HealthService | None: ...

    @property
    def metrics_service(self) -> MetricsService | None: ...

    @property
    def config_admin_service(self) -> ConfigAdminService | None: ...

    @property
    def pdf_diagnostic_service(self) -> PDFDiagnosticService | None: ...

    @property
    def ingestion_service(self) -> IngestionService | None: ...

    @property
    def search_service(self) -> SearchService | None: ...

    @property
    def conversation_service(self) -> ConversationService | None: ...

    @property
    def group_service(self) -> GroupService | None: ...

    @property
    def quality_service(self) -> QualityService | None: ...

    @property
    def benchmark_question_service(self) -> BenchmarkQuestionService | None: ...

    @property
    def benchmark_sweep_service(self) -> BenchmarkSweepService | None: ...

    @property
    def benchmark_sweep_launcher(self) -> BenchmarkSweepLauncher | None: ...

    @property
    def benchmark_run_service(self) -> BenchmarkRunService | None: ...

    @property
    def benchmark_export_service(self) -> BenchmarkExportService | None: ...

    @property
    def benchmark_result_service(self) -> BenchmarkResultService | None: ...

    @property
    def job_service(self) -> JobService | None: ...

    @property
    def chat_log_service(self) -> ChatLogService | None: ...

    @property
    def assignment_service(self) -> AssignmentService | None: ...


class DatabaseServices(Protocol):
    @property
    def uow_factory(self) -> UnitOfWorkFactory | None: ...


class MLServices(Protocol):
    @property
    def preview_cache(self) -> PreviewCache | None: ...


class AuxiliaryServices(Protocol):
    @property
    def api_key_provider(self) -> ApiKeyProviderPort | None: ...

    @property
    def idempotency_store(self) -> IdempotencyStorePort | None: ...

    @property
    def benchmark_history(self) -> BenchmarkHistoryPort | None: ...

    @property
    def cache_invalidator(self) -> CacheInvalidatorPort | None: ...

    @property
    def action_logger(self) -> ActionLoggerPort | None: ...

    @property
    def preview_strategy_factory(self) -> PreviewStrategyFactoryPort | None: ...

    @property
    def job_enqueuer(self) -> JobEnqueuerPort | None: ...

    @property
    def config_masker(self) -> ConfigMaskerPort | None: ...

    @property
    def log_buffer(self) -> LogBufferPort | None: ...


class InfrastructureServices(Protocol):
    @property
    def ml(self) -> MLServices: ...

    @property
    def db(self) -> DatabaseServices: ...

    @property
    def services(self) -> AuxiliaryServices: ...

    @property
    def rate_limit(self) -> RateLimiterPort | None: ...

    @property
    def domain_registry(self) -> DomainProfileRegistry | None: ...

    @property
    def domain_settings(self) -> DomainSettingsPort | None: ...

    @property
    def budget_admin(self) -> BudgetAdminService | None: ...

    @property
    def corpus_control(self) -> CorpusControlPort | None: ...


class ApiContainer(Protocol):
    @property
    def application(self) -> ApplicationServices: ...

    @property
    def infrastructure(self) -> InfrastructureServices: ...
