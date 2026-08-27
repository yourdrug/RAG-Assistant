"""
Depends() providers — thin wrappers that read from ``request.app.state.container``.

All construction happens in ``Container.init()``.  These functions are
only the "glue" between FastAPI's DI system and the pre-built container.
Naming convention: ``create_*`` for all providers (consistent prefix).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from fastapi import Request

if TYPE_CHECKING:
    from application.services.auth_service import AuthService
    from application.services.benchmark_result_service import BenchmarkResultService
    from application.services.benchmark_services import (
        BenchmarkQuestionService,
        BenchmarkRunService,
        BenchmarkSweepService,
    )
    from application.services.chat_log_service import ChatLogService
    from application.services.chat_service import ChatService
    from application.services.chunk_service import ChunkService
    from application.services.config_admin_service import ConfigAdminService
    from application.services.config_service import ConfigService
    from application.services.conversation_service import ConversationService
    from application.services.document_service import DocumentService
    from application.services.group_service import GroupService
    from application.services.health_service import HealthService
    from application.services.ingest_service import IngestAppService
    from application.services.job_service import JobService
    from application.services.metrics_service import MetricsService
    from application.services.pdf_diagnostic_service import PDFDiagnosticService
    from application.services.quality_service import QualityService
    from application.services.search_service import SearchService
    from composition.container import Container
    from infrastructure.auth.api_key_provider import ApiKeyProvider
    from infrastructure.services.ingestion_service import IngestionService

log = logging.getLogger("default")

T = TypeVar("T")


def _create_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Container not initialized — lifespan() did not run")
    return container


def _get_or_raise(value: T | None, name: str) -> T:
    if value is None:
        raise RuntimeError(f"{name} not initialized — Container.init() did not run")
    return value


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def create_ingestion_port(request: Request) -> IngestionService:
    return _get_or_raise(
        _create_container(request).application.ingestion_service, "IngestionService"
    )


def create_preview_cache(request: Request):
    result = _create_container(request).infrastructure.preview_cache
    assert result is not None, "Container not initialized"
    return result


# ---------------------------------------------------------------------------
# Application services
# ---------------------------------------------------------------------------


def create_chat_service(request: Request) -> ChatService:
    return _get_or_raise(_create_container(request).application.chat_service, "ChatService")


def create_auth_service(request: Request) -> AuthService:
    return _get_or_raise(_create_container(request).application.auth_service, "AuthService")


def create_document_service(request: Request) -> DocumentService:
    return _get_or_raise(
        _create_container(request).application.document_service, "DocumentService"
    )


def create_chunk_service(request: Request) -> ChunkService:
    return _get_or_raise(_create_container(request).application.chunk_service, "ChunkService")


def create_ingest_service(request: Request) -> IngestAppService:
    return _get_or_raise(
        _create_container(request).application.ingest_app_service, "IngestAppService"
    )


def create_config_service(request: Request) -> ConfigService:
    return _get_or_raise(_create_container(request).application.config_service, "ConfigService")


def create_health_service(request: Request) -> HealthService:
    return _get_or_raise(_create_container(request).application.health_service, "HealthService")


def create_metrics_service(request: Request) -> MetricsService:
    return _get_or_raise(
        _create_container(request).application.metrics_service, "MetricsService"
    )


def create_config_admin_service(request: Request) -> ConfigAdminService:
    return _get_or_raise(
        _create_container(request).application.config_admin_service, "ConfigAdminService"
    )


def create_pdf_diagnostic_service(request: Request) -> PDFDiagnosticService:
    return _get_or_raise(
        _create_container(request).application.pdf_diagnostic_service, "PDFDiagnosticService"
    )


def create_benchmark_result_service(request: Request) -> BenchmarkResultService:
    return _get_or_raise(
        _create_container(request).application.benchmark_result_service, "BenchmarkResultService"
    )


def create_search_service(request: Request) -> SearchService:
    return _get_or_raise(
        _create_container(request).application.search_service, "SearchService"
    )


def create_conversation_service(request: Request) -> ConversationService:
    return _get_or_raise(
        _create_container(request).application.conversation_service, "ConversationService"
    )


def create_group_service(request: Request) -> GroupService:
    return _get_or_raise(_create_container(request).application.group_service, "GroupService")


def create_quality_service(request: Request) -> QualityService:
    return _get_or_raise(
        _create_container(request).application.quality_service, "QualityService"
    )


def create_benchmark_question_service(request: Request) -> BenchmarkQuestionService:
    return _get_or_raise(
        _create_container(request).application.benchmark_question_service,
        "BenchmarkQuestionService",
    )


def create_benchmark_sweep_service(request: Request) -> BenchmarkSweepService:
    return _get_or_raise(
        _create_container(request).application.benchmark_sweep_service, "BenchmarkSweepService"
    )


def create_benchmark_run_service(request: Request) -> BenchmarkRunService:
    return _get_or_raise(
        _create_container(request).application.benchmark_run_service, "BenchmarkRunService"
    )


def create_job_service(request: Request) -> JobService:
    return _get_or_raise(_create_container(request).application.job_service, "JobService")


def create_chat_log_service(request: Request) -> ChatLogService:
    return _get_or_raise(
        _create_container(request).application.chat_log_service, "ChatLogService"
    )


def create_api_key_provider(request: Request) -> ApiKeyProvider:
    result = _create_container(request).infrastructure.api_key_provider
    assert result is not None, "Container not initialized"
    return result
