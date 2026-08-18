"""DI Container — dataclass-based composition root.

No third-party DI framework.  The entire dependency graph is built in
``Container.init()`` and torn down in ``Container.dispose()``.

Pattern inspired by gipn/backend but implemented with plain dataclasses:

    Container
    ├── infrastructure: InfrastructureContainer
    │   ├── database, uow_factory, ml_clients, ...
    └── application: ApplicationContainer
        ├── chat_service, auth_service, ...

Usage::

    container = Container()
    container.init(database_manager)   # builds all dependencies
    app.state.container = container    # store for request-scoped access
    ...
    await container.dispose()          # cleanup on shutdown
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    from infrastructure.admin.config_admin_adapter import OllamaProbe, QdrantInfo
    from infrastructure.auth.api_key_provider import ApiKeyProvider
    from infrastructure.database.database import DatabaseManager
    from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster
    from infrastructure.events.postgres_config_listener import PostgresConfigListener
    from infrastructure.health.system_health_probe import SystemHealthProbe
    from infrastructure.ml.client_registry import MLClientRegistry
    from infrastructure.ml.langchain_document_parser import (
        LangchainDocumentParser,
        LangchainDocumentSplitter,
    )
    from infrastructure.ml.prometheus_adapter import PrometheusMetricsRegistry
    from infrastructure.ml.summary_adapter import RollingSummaryUpdater
    from infrastructure.repositories.qdrant_vector_store_repository import (
        QdrantVectorStoreRepository,
    )
    from infrastructure.services.benchmark_service import BenchmarkService
    from infrastructure.services.ingestion_service import IngestionService
    from infrastructure.storage import LazyStorage
    from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Infrastructure sub-container
# ---------------------------------------------------------------------------


@dataclass
class InfrastructureContainer:
    """Infrastructure-layer singletons and factories.

    All fields are assigned in ``init()`` — not in ``__init__`` (they use
    ``field(init=False)``).  This keeps the dataclass declarative while
    allowing lazy construction.
    """

    database: DatabaseManager | None = field(default=None)
    config_broadcaster: PostgresConfigBroadcaster | None = field(default=None)
    uow_factory: UnitOfWorkFactory | None = field(default=None)
    vector_store_repo: QdrantVectorStoreRepository | None = field(default=None)
    file_storage: LazyStorage | None = field(default=None)
    document_parser: LangchainDocumentParser | None = field(default=None)
    document_splitter: LangchainDocumentSplitter | None = field(default=None)
    ml_clients: MLClientRegistry | None = field(default=None)
    config_listener: PostgresConfigListener | None = field(default=None)
    health_probe: SystemHealthProbe | None = field(default=None)
    metrics_registry: PrometheusMetricsRegistry | None = field(default=None)
    ollama_probe: OllamaProbe | None = field(default=None)
    qdrant_info: QdrantInfo | None = field(default=None)
    benchmark_service: BenchmarkService | None = field(default=None)
    summary_updater: RollingSummaryUpdater | None = field(default=None)
    api_key_provider: ApiKeyProvider | None = field(default=None)

    def init(self, database_manager: DatabaseManager) -> None:
        """Create all infrastructure-layer objects.

        Raises TypeError if database_manager is not a DatabaseManager.
        """
        from infrastructure.database.database import DatabaseManager

        if not isinstance(database_manager, DatabaseManager):
            raise TypeError(f"Expected DatabaseManager, got {type(database_manager).__name__}")

        from infrastructure.admin.config_admin_adapter import OllamaProbe, QdrantInfo
        from infrastructure.auth.api_key_provider import api_key_provider
        from infrastructure.events.in_process_event_bus import event_bus
        from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster
        from infrastructure.events.postgres_config_listener import PostgresConfigListener
        from infrastructure.health.system_health_probe import SystemHealthProbe
        from infrastructure.ml.client_registry import MLClientRegistry
        from infrastructure.ml.langchain_document_parser import (
            LangchainDocumentParser,
            LangchainDocumentSplitter,
        )
        from infrastructure.ml.prometheus_adapter import PrometheusMetricsRegistry
        from infrastructure.ml.summary_adapter import RollingSummaryUpdater
        from infrastructure.repositories.qdrant_vector_store_repository import (
            QdrantVectorStoreRepository,
        )
        from infrastructure.services.benchmark_service import BenchmarkService
        from infrastructure.storage import LazyStorage
        from infrastructure.uow_factory import UnitOfWorkFactory as ConcreteUoWFactory

        self.database = database_manager
        self.api_key_provider = api_key_provider
        self.config_broadcaster = PostgresConfigBroadcaster(database=database_manager)
        self.uow_factory = ConcreteUoWFactory(
            database=database_manager,
            config_broadcaster=self.config_broadcaster,
        )
        self.vector_store_repo = QdrantVectorStoreRepository()
        self.file_storage = LazyStorage()
        self.document_parser = LangchainDocumentParser()
        self.document_splitter = LangchainDocumentSplitter()
        self.ml_clients = MLClientRegistry()
        self.health_probe = SystemHealthProbe()
        self.metrics_registry = PrometheusMetricsRegistry()
        self.ollama_probe = OllamaProbe()
        self.qdrant_info = QdrantInfo()
        self.benchmark_service = BenchmarkService()
        self.summary_updater = RollingSummaryUpdater()
        self.config_listener = PostgresConfigListener(
            event_bus=event_bus,
            uow_factory=self.uow_factory,
        )

    async def dispose(self) -> None:
        """No infrastructure resources need explicit dispose — lifecycle is
        managed by lifespan().  This method exists for symmetry and future use.
        """


# ---------------------------------------------------------------------------
# Application sub-container
# ---------------------------------------------------------------------------


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
        """Create all application-layer services using infrastructure objects."""
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
        from config import settings
        from infrastructure.auth.jwt_provider import JWTProvider
        from infrastructure.auth.password_hasher import BCryptPasswordHasher
        from infrastructure.events.in_process_event_bus import event_bus
        from infrastructure.ml.pdf_adapter import (
            FitzPDFDocument,
            MLOcrRunner,
            MLPageClassifier,
            MLTextCleaner,
        )
        from infrastructure.ml.rag_service import RagService
        from infrastructure.services.ingestion_service import IngestionService
        from infrastructure.storage import LazyStorage

        uow = infra.uow_factory
        assert uow is not None, "InfrastructureContainer.init() must be called before ApplicationContainer.init()"
        vsr = infra.vector_store_repo
        assert vsr is not None, "InfrastructureContainer.init() must be called before ApplicationContainer.init()"
        fs = infra.file_storage
        assert fs is not None, "InfrastructureContainer.init() must be called before ApplicationContainer.init()"
        ml = infra.ml_clients
        assert ml is not None, "InfrastructureContainer.init() must be called before ApplicationContainer.init()"

        # --- chunk search adapter (bridges UoW.chunks to ChunkSearchPort) ---
        chunk_search = _ChunkSearchAdapter(uow_factory=uow)

        # --- infrastructure-level services ---
        self.ingestion_service = IngestionService(
            vector_store_repo=vsr,
            file_storage=fs,
            uow_factory=uow,
        )

        # --- application services ---
        self.chat_service = ChatService(
            uow_factory=uow,
            rag_service=RagService(ml_clients=ml, chunk_search=chunk_search),
            history_window=settings.history_window,
            rolling_summary_enabled=settings.rolling_summary_enabled,
            summary_updater=infra.summary_updater,
        )
        self.auth_service = AuthService(
            uow_factory=uow,
            password_hasher=BCryptPasswordHasher(),
            token_provider=JWTProvider(),
            api_key_provider=infra.api_key_provider,
        )
        self.document_service = DocumentService(
            uow_factory=uow,
            vector_store_repo=vsr,
            file_storage=fs,
        )
        self.chunk_service = ChunkService(uow_factory=uow, vector_store_repo=vsr)
        self.ingest_app_service = IngestAppService(
            uow_factory=uow,
            ingestion_service=self.ingestion_service,
        )
        self.config_service = ConfigService(uow_factory=uow, event_bus=event_bus)
        self.health_service = HealthService(
            uow_factory=uow,
            probe=infra.health_probe,
            config_listener_provider=infra.config_listener,
            version=settings.version,
            uptime_seconds=settings.uptime_seconds,
            llm_provider=settings.llm_provider,
        )
        self.metrics_service = MetricsService(registry=infra.metrics_registry)
        self.config_admin_service = ConfigAdminService(
            ollama_probe=infra.ollama_probe,
            vectordb_info=infra.qdrant_info,
            openrouter_models_fetcher=_get_openrouter_fetcher(),
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            embed_model=settings.embed_model,
            rerank_model=settings.rerank_model,
            device=settings.resolved_device,
            embed_device=settings.embed_resolved_device,
            rerank_device=settings.embed_resolved_device,
            ocr_engine=settings.ocr_engine,
            ocr_enabled=settings.ocr_enabled,
            openrouter_model=settings.openrouter_model,
            active_collection=settings.collection_name,
        )
        self.pdf_diagnostic_service = PDFDiagnosticService(
            classifier=MLPageClassifier(),
            text_cleaner=MLTextCleaner(),
            ocr=MLOcrRunner(),
            pdf_doc=FitzPDFDocument(),
            storage=LazyStorage(),
        )

        # --- thin domain services ---
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
        if hasattr(self.chat_service, "shutdown"):
            await self.chat_service.shutdown()


# ---------------------------------------------------------------------------
# Main Container
# ---------------------------------------------------------------------------


@dataclass
class Container:
    """Top-level DI container.

    Usage::

        container = Container()
        container.init(database_manager)
        app.state.container = container   # for request-scoped access
        # ... serve requests ...
        await container.dispose()
    """

    infrastructure: InfrastructureContainer = field(default_factory=InfrastructureContainer)
    application: ApplicationContainer = field(default_factory=ApplicationContainer)
    _initialized: bool = field(default=False, repr=False)

    def init(self, database_manager: DatabaseManager) -> None:
        """Build the entire dependency graph.

        Must be called exactly once per process.
        Raises RuntimeError if called more than once.
        """
        if self._initialized:
            raise RuntimeError(
                "Container.init() must be called exactly once. " "Second call detected — this is a bug."
            )
        self.infrastructure.init(database_manager)
        self.application.init(self.infrastructure)
        _subscribe_config_events(self.infrastructure)
        self._initialized = True
        log.info(
            "Container initialized: %d infrastructure + %d application objects",
            len(dataclasses.fields(self.infrastructure)),
            len(dataclasses.fields(self.application)),
        )

    async def dispose(self) -> None:
        """Tear down all resources in reverse order.

        Safe to call even if init() was never called.
        """
        if not self._initialized:
            return
        await self.application.dispose()
        await self.infrastructure.dispose()
        self._initialized = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_openrouter_fetcher():
    """Return the OpenRouter model fetcher function."""
    from infrastructure.admin.config_admin_adapter import fetch_openrouter_models

    return fetch_openrouter_models


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


def _subscribe_config_events(infra: InfrastructureContainer) -> None:
    """Subscribe config-change handlers to the event bus.

    Subscribers that invalidate ML caches use closures over ml_clients,
    so they are created here rather than at module level.
    """
    from domain.events.config_events import ConfigParameterChanged
    from infrastructure.events.in_process_event_bus import event_bus
    from infrastructure.ml.config_subscribers import (
        apply_to_settings,
        audit_log_config_change,
        invalidate_paddle_ocr_cache,
        invalidate_storage_cache,
    )

    bus = event_bus
    ml = infra.ml_clients
    assert ml is not None, "InfrastructureContainer must be initialized before subscribing config events"

    bus.subscribe(ConfigParameterChanged, apply_to_settings)
    bus.subscribe(ConfigParameterChanged, invalidate_paddle_ocr_cache)
    bus.subscribe(ConfigParameterChanged, invalidate_storage_cache)
    bus.subscribe(ConfigParameterChanged, audit_log_config_change)

    # ML-client invalidation via registry (closures over ml_clients)
    def _invalidate_llm(event: ConfigParameterChanged) -> None:
        llm_keys = {
            "llm_provider",
            "llm_model",
            "llm_temperature",
            "llm_top_p",
            "llm_num_ctx_narrow",
            "llm_num_predict_narrow",
            "openrouter_model",
        }
        if event.key in llm_keys:
            ml.invalidate_llm()

    def _invalidate_bm25(event: ConfigParameterChanged) -> None:
        if event.key == "hybrid_enabled":
            ml.invalidate_bm25()

    bus.subscribe(ConfigParameterChanged, _invalidate_llm)
    bus.subscribe(ConfigParameterChanged, _invalidate_bm25)
