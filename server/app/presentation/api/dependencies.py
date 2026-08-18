"""
Depends() providers — thin wrappers that read from ``request.app.state.container``.

All construction happens in ``Container.init()``.  These functions are
only the "glue" between FastAPI's DI system and the pre-built container.
Naming convention: ``create_*`` for all providers (consistent prefix).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.security import APIKeyHeader

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
    from infrastructure.events.postgres_config_listener import PostgresConfigListener
    from infrastructure.ml.langchain_document_parser import (
        LangchainDocumentParser,
        LangchainDocumentSplitter,
    )
    from infrastructure.repositories.qdrant_vector_store_repository import (
        QdrantVectorStoreRepository,
    )
    from infrastructure.services.benchmark_service import BenchmarkService
    from infrastructure.services.ingestion_service import IngestionService
    from infrastructure.storage import LazyStorage
    from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


def _create_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Container not initialized — lifespan() did not run")
    return container


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

auth_key_header = APIKeyHeader(
    name="Authorization",
    auto_error=False,
)


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def create_uow_factory(request: Request) -> UnitOfWorkFactory:
    result = _create_container(request).infrastructure.uow_factory
    assert result is not None, "Container not initialized"
    return result


def create_document_parser(request: Request) -> LangchainDocumentParser:
    result = _create_container(request).infrastructure.document_parser
    assert result is not None, "Container not initialized"
    return result


def create_document_splitter(request: Request) -> LangchainDocumentSplitter:
    result = _create_container(request).infrastructure.document_splitter
    assert result is not None, "Container not initialized"
    return result


def create_vector_store_repo(request: Request) -> QdrantVectorStoreRepository:
    result = _create_container(request).infrastructure.vector_store_repo
    assert result is not None, "Container not initialized"
    return result


def create_file_storage(request: Request) -> LazyStorage:
    result = _create_container(request).infrastructure.file_storage
    assert result is not None, "Container not initialized"
    return result


def create_config_listener(request: Request) -> PostgresConfigListener:
    result = _create_container(request).infrastructure.config_listener
    assert result is not None, "Container not initialized"
    return result


def create_benchmark_service(request: Request) -> BenchmarkService:
    result = _create_container(request).infrastructure.benchmark_service
    assert result is not None, "Container not initialized"
    return result


def create_ingestion_port(request: Request) -> IngestionService:
    return _create_container(request).application.ingestion_service


# ---------------------------------------------------------------------------
# Application services
# ---------------------------------------------------------------------------


def create_chat_service(request: Request) -> ChatService:
    return _create_container(request).application.chat_service


def create_auth_service(request: Request) -> AuthService:
    return _create_container(request).application.auth_service


def create_document_service(request: Request) -> DocumentService:
    return _create_container(request).application.document_service


def create_chunk_service(request: Request) -> ChunkService:
    return _create_container(request).application.chunk_service


def create_ingest_service(request: Request) -> IngestAppService:
    return _create_container(request).application.ingest_app_service


def create_ingestion_service(request: Request) -> IngestionService:
    return _create_container(request).application.ingestion_service


def create_config_service(request: Request) -> ConfigService:
    return _create_container(request).application.config_service


def create_health_service(request: Request) -> HealthService:
    return _create_container(request).application.health_service


def create_metrics_service(request: Request) -> MetricsService:
    return _create_container(request).application.metrics_service


def create_config_admin_service(request: Request) -> ConfigAdminService:
    return _create_container(request).application.config_admin_service


def create_pdf_diagnostic_service(request: Request) -> PDFDiagnosticService:
    return _create_container(request).application.pdf_diagnostic_service


def create_benchmark_result_service(request: Request) -> BenchmarkResultService:
    return _create_container(request).application.benchmark_result_service


def create_search_service(request: Request) -> SearchService:
    return _create_container(request).application.search_service


def create_conversation_service(request: Request) -> ConversationService:
    return _create_container(request).application.conversation_service


def create_group_service(request: Request) -> GroupService:
    return _create_container(request).application.group_service


def create_quality_service(request: Request) -> QualityService:
    return _create_container(request).application.quality_service


def create_benchmark_question_service(request: Request) -> BenchmarkQuestionService:
    return _create_container(request).application.benchmark_question_service


def create_benchmark_sweep_service(request: Request) -> BenchmarkSweepService:
    return _create_container(request).application.benchmark_sweep_service


def create_benchmark_run_service(request: Request) -> BenchmarkRunService:
    return _create_container(request).application.benchmark_run_service


def create_job_service(request: Request) -> JobService:
    return _create_container(request).application.job_service


def create_chat_log_service(request: Request) -> ChatLogService:
    return _create_container(request).application.chat_log_service


def create_api_key_provider(request: Request) -> ApiKeyProvider:
    result = _create_container(request).infrastructure.api_key_provider
    assert result is not None, "Container not initialized"
    return result
