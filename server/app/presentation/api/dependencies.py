"""Depends() providers — thin wrappers that read from ``request.app.state.container``.

All construction happens in ``Container.init()``.  These functions are
only the "glue" between FastAPI's DI system and the pre-built container.
Naming convention: ``create_*`` for all providers (consistent prefix).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import Depends, HTTPException, Request

from application.act_versioning import ActVersioningService
from application.admin import ConfigMaskerPort, CorpusControlPort, LogBufferPort
from application.benchmark import BenchmarkExportService, BenchmarkHistoryPort
from application.budget import BudgetAdminService
from application.documents import DocumentDeletionJobs, PreviewCache, PreviewStrategyFactoryPort
from application.jobs.service import JobService
from application.ports.action_logger import ActionLoggerPort
from application.ports.cache_invalidator import CacheInvalidatorPort
from application.ports.idempotency_store import IdempotencyStorePort
from application.ports.job_enqueuer import JobEnqueuerPort
from application.ports.route_configs import BenchmarkConfig, CacheConfig, StorageConfig, UploadConfig
from config import settings
from domain.ingestion.ports import DomainSettingsPort
from domain.ingestion.profiles.registry import DomainProfileRegistry
from presentation.api.runtime import ApiContainer


if TYPE_CHECKING:
    from application.admin import ConfigAdminService, ConfigService
    from application.admin.metrics import MetricsService
    from application.benchmark import (
        BenchmarkQuestionService,
        BenchmarkResultService,
        BenchmarkRunService,
        BenchmarkSweepLauncher,
        BenchmarkSweepService,
    )
    from application.chat import ChatLogService, ChatService, ConversationService
    from application.chunks import ChunkMutationService, ChunkQueryService
    from application.documents import DocumentCommandService, DocumentQueryService, ManualDocumentService
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
    from application.ports.rate_limit import RateLimiterPort
    from application.search import SearchService

log = logging.getLogger("default")

IDEMPOTENCY_KEY_MAX_LENGTH = 200


def create_container(request: Request) -> ApiContainer:
    container = getattr(request.app.state, "container", None)

    if container is None:
        message = "Container not initialized — lifespan() did not run"
        raise RuntimeError(message)

    return cast("ApiContainer", container)


def get_or_raise[T](value: T | None, name: str) -> T:
    if value is None:
        message = f"{name} not initialized — Container.init() did not run"
        raise RuntimeError(message)

    return value


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def create_ingestion_port(request: Request) -> IngestionService:
    return get_or_raise(create_container(request).application.ingestion_service, "IngestionService")


def create_preview_cache(request: Request) -> PreviewCache:
    return get_or_raise(create_container(request).infrastructure.ml.preview_cache, "PreviewCache")


# ---------------------------------------------------------------------------
# Application services
# ---------------------------------------------------------------------------


def create_chat_service(request: Request) -> ChatService:
    return get_or_raise(create_container(request).application.chat_service, "ChatService")


def create_auth_service(request: Request) -> AuthService:
    return get_or_raise(create_container(request).application.auth_service, "AuthService")


def create_user_management_service(request: Request) -> UserManagementService:
    return get_or_raise(
        create_container(request).application.user_management_service, "UserManagementService"
    )


def create_api_key_service(request: Request) -> ApiKeyService:
    return get_or_raise(create_container(request).application.api_key_service, "ApiKeyService")


def create_document_command_service(request: Request) -> DocumentCommandService:
    return get_or_raise(
        create_container(request).application.document_command_service, "DocumentCommandService"
    )


def create_document_query_service(request: Request) -> DocumentQueryService:
    return get_or_raise(create_container(request).application.document_query_service, "DocumentQueryService")


def create_chunk_query_service(request: Request) -> ChunkQueryService:
    return get_or_raise(create_container(request).application.chunk_query_service, "ChunkQueryService")


def create_chunk_mutation_service(request: Request) -> ChunkMutationService:
    return get_or_raise(
        create_container(request).application.chunk_mutation_service,
        "ChunkMutationService",
    )


def create_manual_document_service(request: Request) -> ManualDocumentService:
    return get_or_raise(
        create_container(request).application.manual_document_service,
        "ManualDocumentService",
    )


def create_ingest_service(request: Request) -> IngestionService:
    return get_or_raise(create_container(request).application.ingestion_service, "IngestionService")


def create_config_service(request: Request) -> ConfigService:
    return get_or_raise(create_container(request).application.config_service, "ConfigService")


def create_health_service(request: Request) -> HealthService:
    return get_or_raise(create_container(request).application.health_service, "HealthService")


def create_metrics_service(request: Request) -> MetricsService:
    return get_or_raise(create_container(request).application.metrics_service, "MetricsService")


def create_config_admin_service(request: Request) -> ConfigAdminService:
    return get_or_raise(create_container(request).application.config_admin_service, "ConfigAdminService")


def create_pdf_diagnostic_service(request: Request) -> PDFDiagnosticService:
    return get_or_raise(create_container(request).application.pdf_diagnostic_service, "PDFDiagnosticService")


def create_benchmark_result_service(request: Request) -> BenchmarkResultService:
    return get_or_raise(
        create_container(request).application.benchmark_result_service, "BenchmarkResultService"
    )


def create_search_service(request: Request) -> SearchService:
    return get_or_raise(create_container(request).application.search_service, "SearchService")


def create_conversation_service(request: Request) -> ConversationService:
    return get_or_raise(create_container(request).application.conversation_service, "ConversationService")


def create_group_service(request: Request) -> GroupService:
    return get_or_raise(create_container(request).application.group_service, "GroupService")


def create_assignment_service(request: Request) -> AssignmentService:
    return get_or_raise(create_container(request).application.assignment_service, "AssignmentService")


def create_quality_service(request: Request) -> QualityService:
    return get_or_raise(create_container(request).application.quality_service, "QualityService")


def create_benchmark_question_service(request: Request) -> BenchmarkQuestionService:
    return get_or_raise(
        create_container(request).application.benchmark_question_service,
        "BenchmarkQuestionService",
    )


def create_benchmark_sweep_service(request: Request) -> BenchmarkSweepService:
    return get_or_raise(
        create_container(request).application.benchmark_sweep_service, "BenchmarkSweepService"
    )


def create_benchmark_sweep_launcher(request: Request) -> BenchmarkSweepLauncher:
    return get_or_raise(
        create_container(request).application.benchmark_sweep_launcher, "BenchmarkSweepLauncher"
    )


def create_benchmark_run_service(request: Request) -> BenchmarkRunService:
    return get_or_raise(create_container(request).application.benchmark_run_service, "BenchmarkRunService")


def create_job_service(request: Request) -> JobService:
    return get_or_raise(create_container(request).application.job_service, "JobService")


def create_chat_log_service(request: Request) -> ChatLogService:
    return get_or_raise(create_container(request).application.chat_log_service, "ChatLogService")


def create_api_key_provider(request: Request) -> ApiKeyProviderPort:
    return get_or_raise(create_container(request).infrastructure.services.api_key_provider, "ApiKeyProvider")


def create_rate_limiter(request: Request) -> RateLimiterPort | None:
    """Rate limiter port; None when rate limiting is disabled by settings."""
    return create_container(request).infrastructure.rate_limit


def get_idempotency_key(request: Request) -> str | None:
    """Extract Idempotency-Key header. None when not provided."""
    key = request.headers.get("Idempotency-Key")

    if key is None:
        return None

    key = key.strip()

    if not key or len(key) > IDEMPOTENCY_KEY_MAX_LENGTH:
        raise HTTPException(status_code=400, detail="Idempotency-Key must contain 1 to 200 characters")

    return key


def create_idempotency_store(request: Request) -> IdempotencyStorePort:
    """Create IdempotencyStore backed by Redis."""
    return get_or_raise(
        create_container(request).infrastructure.services.idempotency_store, "IdempotencyStore"
    )


def create_benchmark_history_port(request: Request) -> BenchmarkHistoryPort:
    return get_or_raise(
        create_container(request).infrastructure.services.benchmark_history, "BenchmarkHistoryPort"
    )


def create_act_versioning_service(request: Request) -> ActVersioningService:

    c = create_container(request)
    domain_settings = c.infrastructure.domain_settings

    if domain_settings is None:
        message = "domain_settings not initialized"
        raise ValueError(message)

    return ActVersioningService(
        uow_factory=get_or_raise(c.infrastructure.db.uow_factory, "uow_factory"),
        settings=domain_settings,
        cache_invalidator=c.infrastructure.services.cache_invalidator,
    )


def create_domain_registry(request: Request) -> DomainProfileRegistry | None:
    c = create_container(request)
    return c.infrastructure.domain_registry


def create_domain_settings(request: Request) -> DomainSettingsPort | None:
    c = create_container(request)
    return c.infrastructure.domain_settings


# ---------------------------------------------------------------------------
# Presentation-layer adapters (infrastructure ports)
# ---------------------------------------------------------------------------


def create_action_logger(request: Request) -> ActionLoggerPort:
    return get_or_raise(create_container(request).infrastructure.services.action_logger, "ActionLoggerPort")


def create_cache_invalidator(request: Request) -> CacheInvalidatorPort:
    return get_or_raise(
        create_container(request).infrastructure.services.cache_invalidator, "CacheInvalidatorPort"
    )


def create_preview_strategy_factory(request: Request) -> PreviewStrategyFactoryPort:
    return get_or_raise(
        create_container(request).infrastructure.services.preview_strategy_factory, "PreviewStrategyFactory"
    )


def create_job_enqueuer(request: Request) -> JobEnqueuerPort:
    return get_or_raise(create_container(request).infrastructure.services.job_enqueuer, "JobEnqueuerPort")


def create_config_masker(request: Request) -> ConfigMaskerPort:
    return get_or_raise(create_container(request).infrastructure.services.config_masker, "ConfigMaskerPort")


def create_log_buffer(request: Request) -> LogBufferPort:
    return get_or_raise(create_container(request).infrastructure.services.log_buffer, "LogBufferPort")


# ---------------------------------------------------------------------------
# Route config dataclasses (typed, immutable config for presentation layer)
# ---------------------------------------------------------------------------


def create_upload_config(_request: Request) -> UploadConfig:

    return UploadConfig(max_upload_size_mb=settings.max_upload_size_mb)


def create_benchmark_config(_request: Request) -> BenchmarkConfig:

    return BenchmarkConfig(
        data_dir=settings.data_dir,
        retriever_top_k=settings.retriever_top_k,
        llm_model=settings.openrouter_model,
    )


def create_cache_config(_request: Request) -> CacheConfig:

    return CacheConfig(cache_enabled=settings.cache_enabled)


def create_storage_config(_request: Request) -> StorageConfig:

    return StorageConfig(redis_url=settings.redis_url)


def create_document_deletion_jobs(request: Request) -> DocumentDeletionJobs:
    return get_or_raise(create_container(request).application.document_deletion_jobs, "DocumentDeletionJobs")


def create_budget_admin_service(request: Request) -> BudgetAdminService:
    return get_or_raise(create_container(request).infrastructure.budget_admin, "BudgetAdminService")


def create_corpus_control(request: Request) -> CorpusControlPort:
    return get_or_raise(create_container(request).infrastructure.corpus_control, "CorpusControl")


def create_benchmark_export_service(request: Request) -> BenchmarkExportService:
    return get_or_raise(
        create_container(request).application.benchmark_export_service, "BenchmarkExportService"
    )


@dataclass(frozen=True)
class JobSubmission:
    jobs: JobService
    enqueuer: JobEnqueuerPort
    log: ActionLoggerPort


def create_job_submission(
    jobs: Annotated[JobService, Depends(create_job_service)],
    enqueuer: Annotated[JobEnqueuerPort, Depends(create_job_enqueuer)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> JobSubmission:
    return JobSubmission(jobs, enqueuer, log)


@dataclass(frozen=True)
class IdempotencyContext:
    key: str | None
    store: IdempotencyStorePort


def create_idempotency_context(
    key: Annotated[str | None, Depends(get_idempotency_key)],
    store: Annotated[IdempotencyStorePort, Depends(create_idempotency_store)],
) -> IdempotencyContext:
    return IdempotencyContext(key, store)
