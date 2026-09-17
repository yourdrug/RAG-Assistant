"""Application sub-container — wired application-layer services."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from composition.utils import _missing_fields, _require
from composition.service_providers import create_ingestion_service
from config import settings
from infrastructure.adapters.chunk_search_adapter import ChunkSearchAdapter
from infrastructure.bm25.bm25_updater import BM25IndexAdapter

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
    from application.services.assignment_service import AssignmentService
    from application.services.document_service import DocumentService
    from application.services.group_service import GroupService
    from application.services.health_service import HealthService
    from application.services.ingest_service import IngestAppService
    from application.services.job_service import JobService
    from application.services.metrics_service import MetricsService
    from application.services.pdf_diagnostic_service import PDFDiagnosticService
    from application.services.quality_service import QualityService
    from application.services.search_service import SearchService
    from infrastructure.ml.rag_service import RagService
    from application.services.ingestion_orchestrator import IngestionService

    from composition.infrastructure import InfrastructureContainer

log = logging.getLogger("default")


def _get_openrouter_fetcher():
    """Return the OpenRouter model fetcher function."""
    from infrastructure.ml.clients.factories import fetch_openrouter_models

    return fetch_openrouter_models


def _make_act_versioning(infra: "InfrastructureContainer", uow_factory):
    """Create ActVersioningService if domain registry and settings are available."""
    if infra.domain_registry is None or infra.domain_settings is None:
        return None
    from application.services.act_versioning_service import ActVersioningService

    return ActVersioningService(uow_factory=uow_factory, settings=infra.domain_settings)


@dataclass
class ApplicationContainer:
    """Application-layer services — all wired via constructor injection.

    All fields are assigned in ``init()``.
    """

    rag_service: RagService | None = field(default=None)
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
    assignment_service: AssignmentService | None = field(default=None)

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
        from application.services.assignment_service import AssignmentService
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
        from infrastructure.ml.config.settings_adapters import (
            LiveChatSettings,
            LiveChunkSettings,
            LiveConfigAdminSettings,
            LiveHealthSettings,
        )
        from infrastructure.adapters.pii_redactor_adapter import PIIRedactorAdapter
        from infrastructure.ml.guardrails.pdf_adapter import (
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

        chunk_search = ChunkSearchAdapter(uow_factory=uow)
        self.rag_service = RagService(
            ml_clients=ml,
            chunk_search=chunk_search,
            domain_registry=infra.domain_registry,
        )

        self.ingestion_service = _require(
            create_ingestion_service(infra, uow_factory=uow),
            "ingestion_service",
        )

        summary_updater = _require(infra.summary_updater, "summary_updater")
        api_key_provider = _require(infra.api_key_provider, "api_key_provider")
        health_probe = _require(infra.health_probe, "health_probe")
        config_listener = _require(infra.config_listener, "config_listener")
        metrics_registry = _require(infra.metrics_registry, "metrics_registry")
        ollama_probe = _require(infra.ollama_probe, "ollama_probe")
        qdrant_info = _require(infra.qdrant_info, "qdrant_info")

        self.chat_log_service = ChatLogService(uow_factory=uow)
        self.conversation_service = ConversationService(
            uow_factory=uow,
            summary_updater=summary_updater,
            chat_settings=LiveChatSettings(),
        )

        self.chat_service = ChatService(
            uow_factory=uow,
            rag_service=self.rag_service,
            chat_settings=LiveChatSettings(),
            chat_log_service=self.chat_log_service,
            conversation_service=self.conversation_service,
            pii_redactor=PIIRedactorAdapter(pii_redaction_enabled=settings.pii_redaction_enabled),
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
            bm25_index=BM25IndexAdapter(ml),
            domain_registry=infra.domain_registry,
            act_versioning_service=_make_act_versioning(infra, uow),
        )
        self.chunk_service = ChunkService(
            uow_factory=uow,
            vector_store_repo=vsr,
            chunk_settings=LiveChunkSettings(),
            bm25_index=BM25IndexAdapter(ml),
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
        self.group_service = GroupService(uow_factory=uow)
        self.assignment_service = AssignmentService(uow_factory=uow)
        self.quality_service = QualityService(uow_factory=uow)
        self.benchmark_question_service = BenchmarkQuestionService(uow_factory=uow)
        self.benchmark_sweep_service = BenchmarkSweepService(uow_factory=uow)
        self.benchmark_run_service = BenchmarkRunService(uow_factory=uow)
        self.benchmark_result_service = BenchmarkResultService(uow_factory=uow)
        self.job_service = JobService(uow_factory=uow)

        # Wire rag_service into benchmark_service for full-pipeline benchmarking
        if infra.benchmark_service is not None:
            infra.benchmark_service.set_rag_service(self.rag_service)

    def validate(self) -> list[str]:
        """Return names of fields that are still ``None`` after init()."""
        return _missing_fields(self)

    async def dispose(self) -> None:
        """Shutdown all application services that have explicit shutdown methods.

        Iterates over all fields — any service with a ``shutdown()`` method
        gets called.  Failures are logged but do not prevent remaining
        services from being cleaned up.
        """
        import dataclasses

        for f in dataclasses.fields(self):
            svc = getattr(self, f.name, None)
            if svc is not None and hasattr(svc, "shutdown"):
                try:
                    await svc.shutdown()
                except Exception:
                    log.warning("Failed to shutdown %s", f.name, exc_info=True)
