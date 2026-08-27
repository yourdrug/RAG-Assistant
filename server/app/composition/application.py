"""Application sub-container — wired application-layer services."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

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
    from infrastructure.services.ingestion_service import IngestionService
    from infrastructure.uow_factory import UnitOfWorkFactory

    from composition.infrastructure import InfrastructureContainer

T = TypeVar("T")

log = logging.getLogger("default")


def _get_openrouter_fetcher():
    """Return the OpenRouter model fetcher function."""
    from infrastructure.admin.config_admin_adapter import fetch_openrouter_models

    return fetch_openrouter_models


def _require(value: T | None, name: str) -> T:
    if value is None:
        raise RuntimeError(f"{name} not initialized — InfrastructureContainer.init() must be called first")
    return value


class _ChunkSearchAdapter:
    """Bridges UoW.chunks to the ChunkSearchPort expected by RagService."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def search_substring(self, query, user, group_ids, limit=20, mode="exact"):
        async with self._uow_factory.create() as uow:
            return await uow.chunks.search_substring(
                query=query,
                user=user,
                group_ids=group_ids,
                limit=limit,
                mode=mode,
            )


@dataclass
class ApplicationContainer:
    """Application-layer services — all wired via constructor injection.

    All fields are assigned in ``init()``.
    """

    chat_service: ChatService | None = field(default=None)
    auth_service: AuthService | None = field(default=None)
    document_service: DocumentService | None = field(default=None)
    chunk_service: ChunkService | None = field(default=None)
    ingest_app_service: IngestAppService | None = field(default=None)
    config_service: ConfigService | None = field(default=None)
    health_service: HealthService | None = field(default=None)
    metrics_service: MetricsService | None = field(default=None)
    config_admin_service: ConfigAdminService | None = field(default=None)
    pdf_diagnostic_service: PDFDiagnosticService | None = field(default=None)
    ingestion_service: IngestionService | None = field(default=None)
    search_service: SearchService | None = field(default=None)
    conversation_service: ConversationService | None = field(default=None)
    group_service: GroupService | None = field(default=None)
    quality_service: QualityService | None = field(default=None)
    benchmark_question_service: BenchmarkQuestionService | None = field(default=None)
    benchmark_sweep_service: BenchmarkSweepService | None = field(default=None)
    benchmark_run_service: BenchmarkRunService | None = field(default=None)
    benchmark_result_service: BenchmarkResultService | None = field(default=None)
    job_service: JobService | None = field(default=None)
    chat_log_service: ChatLogService | None = field(default=None)

    def init(self, infra: InfrastructureContainer) -> None:
        """Create all application-layer services using infrastructure objects.

        Raises RuntimeError if InfrastructureContainer has not been initialized.
        """
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
        from infrastructure.auth.jwt_provider import JWTProvider
        from infrastructure.auth.password_hasher import BCryptPasswordHasher
        from infrastructure.events.in_process_event_bus import event_bus
        from infrastructure.ml.settings_adapters import (
            LiveChatSettings,
            LiveChunkSettings,
            LiveConfigAdminSettings,
            LiveHealthSettings,
        )
        from infrastructure.ml.pdf_adapter import (
            FitzPDFDocument,
            MLOcrRunner,
            MLPageClassifier,
            MLTextCleaner,
        )
        from infrastructure.ml.rag_service import RagService

        uow = _require(infra.uow_factory, "uow_factory")
        vsr = _require(infra.vector_store_repo, "vector_store_repo")
        fs = _require(infra.file_storage, "file_storage")
        ml = _require(infra.ml_clients, "ml_clients")

        chunk_search = _ChunkSearchAdapter(uow_factory=uow)

        self.ingestion_service = infra.create_ingestion_service(uow_factory=uow)
        assert self.ingestion_service is not None

        summary_updater = _require(infra.summary_updater, "summary_updater")
        api_key_provider = _require(infra.api_key_provider, "api_key_provider")
        health_probe = _require(infra.health_probe, "health_probe")
        config_listener = _require(infra.config_listener, "config_listener")
        metrics_registry = _require(infra.metrics_registry, "metrics_registry")
        ollama_probe = _require(infra.ollama_probe, "ollama_probe")
        qdrant_info = _require(infra.qdrant_info, "qdrant_info")

        self.chat_service = ChatService(
            uow_factory=uow,
            rag_service=RagService(ml_clients=ml, chunk_search=chunk_search),
            chat_settings=LiveChatSettings(),
            summary_updater=summary_updater,
        )
        self.auth_service = AuthService(
            uow_factory=uow,
            password_hasher=BCryptPasswordHasher(),
            token_provider=JWTProvider(),
            api_key_provider=api_key_provider,
        )
        self.document_service = DocumentService(
            uow_factory=uow,
            vector_store_repo=vsr,
            file_storage=fs,
        )
        self.chunk_service = ChunkService(
            uow_factory=uow, vector_store_repo=vsr, chunk_settings=LiveChunkSettings()
        )
        self.ingest_app_service = IngestAppService(
            uow_factory=uow,
            ingestion_service=self.ingestion_service,
        )
        self.config_service = ConfigService(uow_factory=uow, event_bus=event_bus)
        self.health_service = HealthService(
            uow_factory=uow,
            probe=health_probe,
            config_listener_provider=config_listener,
            health_settings=LiveHealthSettings(),
        )
        self.metrics_service = MetricsService(registry=metrics_registry)
        self.config_admin_service = ConfigAdminService(
            ollama_probe=ollama_probe,
            vectordb_info=qdrant_info,
            admin_settings=LiveConfigAdminSettings(),
            openrouter_models_fetcher=_get_openrouter_fetcher(),
        )
        self.pdf_diagnostic_service = PDFDiagnosticService(
            classifier=MLPageClassifier(),
            text_cleaner=MLTextCleaner(),
            ocr=MLOcrRunner(),
            pdf_doc=FitzPDFDocument(),
            storage=fs,
            preview_cache=infra.preview_cache,
        )

        self.search_service = SearchService(uow_factory=uow)
        self.conversation_service = ConversationService(uow_factory=uow)
        self.group_service = GroupService(uow_factory=uow)
        self.quality_service = QualityService(uow_factory=uow)
        self.benchmark_question_service = BenchmarkQuestionService(uow_factory=uow)
        self.benchmark_sweep_service = BenchmarkSweepService(uow_factory=uow)
        self.benchmark_run_service = BenchmarkRunService(uow_factory=uow)
        self.benchmark_result_service = BenchmarkResultService(uow_factory=uow)
        self.job_service = JobService(uow_factory=uow)
        self.chat_log_service = ChatLogService(uow_factory=uow)

    async def dispose(self) -> None:
        """Shutdown application services that have explicit shutdown methods."""
        if self.chat_service is not None and hasattr(self.chat_service, "shutdown"):
            await self.chat_service.shutdown()
