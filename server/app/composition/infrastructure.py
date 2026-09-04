"""Infrastructure sub-container — singletons for database, ML, storage, etc.

Organized into sub-containers for Single Responsibility:
  - DatabaseContainer: DB connection, UoW factory, config broadcaster
  - MLContainer: ML clients, vector store, adapters
  - EventContainer: config listener, outbox listener/dispatcher
  - InfrastructureContainer: top-level aggregator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from composition._utils import _missing_fields, _require

if TYPE_CHECKING:
    from application.services.preview_cache import PreviewCache
    from infrastructure.admin.config_admin_adapter import OllamaProbe, QdrantInfo
    from infrastructure.auth.api_key_provider import ApiKeyProvider
    from infrastructure.database.database import DatabaseManager
    from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster
    from infrastructure.events.postgres_config_listener import PostgresConfigListener
    from infrastructure.health.system_health_probe import SystemHealthProbe
    from infrastructure.ml.client_registry import MLClientRegistry
    from infrastructure.ml.extraction_adapter import MLContentExtractor, MLPDFQualityAssessor
    from infrastructure.ml.langchain_document_parser import (
        LangchainDocumentParser,
        LangchainDocumentSplitter,
    )
    from infrastructure.ml.metrics_adapter import PrometheusMetricsCollector
    from infrastructure.ml.prometheus_adapter import PrometheusMetricsRegistry
    from infrastructure.ml.summary_adapter import RollingSummaryUpdater
    from infrastructure.repositories.qdrant_vector_store_repository import (
        QdrantVectorStoreRepository,
    )
    from infrastructure.services.benchmark_service import BenchmarkService
    from infrastructure.storage import LazyStorage
    from infrastructure.uow_factory import UnitOfWorkFactory
    from infrastructure.outbox_dispatcher import OutboxDispatcher
    from infrastructure.events.postgres_outbox_listener import PostgresOutboxListener

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Sub-containers
# ---------------------------------------------------------------------------


@dataclass
class DatabaseContainer:
    """Database-layer singletons: connection, UoW, config broadcasting."""

    database: DatabaseManager | None = field(default=None)
    uow_factory: UnitOfWorkFactory | None = field(default=None)
    config_broadcaster: PostgresConfigBroadcaster | None = field(default=None)

    def init(self, database_manager: DatabaseManager) -> None:
        from infrastructure.database.database import DatabaseManager as DBM
        from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster
        from infrastructure.uow_factory import UnitOfWorkFactory

        if not isinstance(database_manager, DBM):
            raise TypeError(f"Expected DatabaseManager, got {type(database_manager).__name__}")

        self.database = database_manager
        self.config_broadcaster = PostgresConfigBroadcaster()
        self.uow_factory = UnitOfWorkFactory(
            database=database_manager,
            config_broadcaster=self.config_broadcaster,
        )

    @property
    def db(self) -> DatabaseManager:
        return _require(self.database, "database")

    @property
    def uow(self) -> UnitOfWorkFactory:
        return _require(self.uow_factory, "uow_factory")


@dataclass
class MLContainer:
    """ML-layer singletons: clients, vector store, adapters."""

    ml_clients: MLClientRegistry | None = field(default=None)
    vector_store_repo: QdrantVectorStoreRepository | None = field(default=None)
    file_storage: LazyStorage | None = field(default=None)
    document_parser: LangchainDocumentParser | None = field(default=None)
    document_splitter: LangchainDocumentSplitter | None = field(default=None)
    summary_updater: RollingSummaryUpdater | None = field(default=None)
    content_extractor: MLContentExtractor | None = field(default=None)
    pdf_quality_assessor: MLPDFQualityAssessor | None = field(default=None)
    metrics_collector: PrometheusMetricsCollector | None = field(default=None)
    metrics_registry: PrometheusMetricsRegistry | None = field(default=None)
    preview_cache: PreviewCache | None = field(default=None)
    benchmark_service: BenchmarkService | None = field(default=None)

    def init(self, uow_factory: UnitOfWorkFactory) -> None:
        from infrastructure.ml.client_registry import MLClientRegistry
        from infrastructure.ml.extraction_adapter import MLContentExtractor, MLPDFQualityAssessor
        from infrastructure.ml.langchain_document_parser import (
            LangchainDocumentParser,
            LangchainDocumentSplitter,
        )
        from infrastructure.ml.metrics_adapter import PrometheusMetricsCollector
        from infrastructure.ml.prometheus_adapter import PrometheusMetricsRegistry
        from infrastructure.ml.summary_adapter import RollingSummaryUpdater
        from infrastructure.repositories.qdrant_vector_store_repository import (
            QdrantVectorStoreRepository,
        )
        from infrastructure.services.benchmark_service import BenchmarkService
        from infrastructure.storage import LazyStorage
        from application.services.preview_cache import PreviewCache

        self.ml_clients = MLClientRegistry()
        self.vector_store_repo = QdrantVectorStoreRepository(ml_clients=self.ml_clients)
        self.file_storage = LazyStorage()
        self.preview_cache = PreviewCache(storage=self.file_storage)
        self.document_parser = LangchainDocumentParser()
        self.document_splitter = LangchainDocumentSplitter()
        self.metrics_registry = PrometheusMetricsRegistry()
        self.benchmark_service = BenchmarkService()
        self.summary_updater = RollingSummaryUpdater(ml_clients=self.ml_clients)
        self.content_extractor = MLContentExtractor()
        self.pdf_quality_assessor = MLPDFQualityAssessor()
        self.metrics_collector = PrometheusMetricsCollector()

    def dispose(self) -> None:
        """Clear ML-specific caches and release resources."""
        if self.file_storage is not None:
            self.file_storage.clear_cache()

        # Clear OCR/lru_cache caches — import may fail if optional deps
        # (paddleocr, surya) are not installed.
        try:
            from infrastructure.ml.ingestion import _get_paddle_ocr, _get_surya_predictors
        except ImportError:
            return
        try:
            _get_paddle_ocr.cache_clear()
            _get_surya_predictors.cache_clear()
        except Exception:
            log.warning("Failed to clear OCR caches", exc_info=True)

    @property
    def clients(self) -> MLClientRegistry:
        return _require(self.ml_clients, "ml_clients")

    @property
    def vector_store(self) -> QdrantVectorStoreRepository:
        return _require(self.vector_store_repo, "vector_store_repo")

    @property
    def storage(self) -> LazyStorage:
        return _require(self.file_storage, "file_storage")


@dataclass
class EventContainer:
    """Event-layer singletons: config listener, outbox dispatcher/listener."""

    config_listener: PostgresConfigListener | None = field(default=None)
    outbox_dispatcher: OutboxDispatcher | None = field(default=None)
    outbox_listener: PostgresOutboxListener | None = field(default=None)

    def init(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: QdrantVectorStoreRepository,
        event_bus,
    ) -> None:
        from infrastructure.events.postgres_config_listener import PostgresConfigListener
        from infrastructure.outbox_dispatcher import OutboxDispatcher
        from infrastructure.events.postgres_outbox_listener import PostgresOutboxListener
        from config import settings as app_settings

        self.config_listener = PostgresConfigListener(
            event_bus=event_bus,
            uow_factory=uow_factory,
        )
        self.outbox_dispatcher = OutboxDispatcher(
            uow_factory=uow_factory,
            vector_store=vector_store_repo,
        )
        self.outbox_listener = PostgresOutboxListener(
            dispatcher=self.outbox_dispatcher,
            db_config={
                "db_host": app_settings.db_host,
                "db_port": app_settings.db_port,
                "db_user": app_settings.db_user,
                "db_password": app_settings.db_password,
                "db_name": app_settings.db_name,
            },
        )

    async def dispose(self) -> None:
        """Stop listeners and close dispatcher."""
        if self.config_listener is not None:
            await self.config_listener.stop()
        if self.outbox_listener is not None:
            await self.outbox_listener.stop()


@dataclass
class ServiceContainer:
    """Auxiliary infrastructure services: health, admin, auth."""

    health_probe: SystemHealthProbe | None = field(default=None)
    ollama_probe: OllamaProbe | None = field(default=None)
    qdrant_info: QdrantInfo | None = field(default=None)
    api_key_provider: ApiKeyProvider | None = field(default=None)

    def init(self) -> None:
        from infrastructure.admin.config_admin_adapter import OllamaProbe, QdrantInfo
        from infrastructure.auth.api_key_provider import api_key_provider
        from infrastructure.health.system_health_probe import SystemHealthProbe

        self.health_probe = SystemHealthProbe()
        self.ollama_probe = OllamaProbe()
        self.qdrant_info = QdrantInfo()
        self.api_key_provider = api_key_provider

    @property
    def health(self) -> SystemHealthProbe:
        return _require(self.health_probe, "health_probe")

    @property
    def ollama(self) -> OllamaProbe:
        return _require(self.ollama_probe, "ollama_probe")

    @property
    def vectordb(self) -> QdrantInfo:
        return _require(self.qdrant_info, "qdrant_info")

    @property
    def api_keys(self) -> ApiKeyProvider:
        return _require(self.api_key_provider, "api_key_provider")


# ---------------------------------------------------------------------------
# Top-level Infrastructure Container
# ---------------------------------------------------------------------------


@dataclass
class InfrastructureContainer:
    """Top-level infrastructure container — aggregates sub-containers.

    Usage::

        infra = InfrastructureContainer()
        infra.init(database_manager)
        # ... use infra.db, infra.ml, etc. ...
        await infra.dispose()
    """

    db: DatabaseContainer = field(default_factory=DatabaseContainer)
    ml: MLContainer = field(default_factory=MLContainer)
    events: EventContainer = field(default_factory=EventContainer)
    services: ServiceContainer = field(default_factory=ServiceContainer)
    _initialized: bool = field(default=False, repr=False)

    def init(self, database_manager: DatabaseManager) -> None:
        """Create all infrastructure-layer objects.

        Order matters: DB → ML → Services → Events (events depend on others).
        """
        from infrastructure.events.in_process_event_bus import event_bus

        self.db.init(database_manager)
        self.ml.init(uow_factory=self.db.uow)
        self.services.init()
        self.events.init(
            uow_factory=self.db.uow,
            vector_store_repo=self.ml.vector_store,
            event_bus=event_bus,
        )
        self._initialized = True
        log.info("Infrastructure container initialized")

    def validate(self) -> list[str]:
        """Return list of missing dependency names (empty = all good).

        Uses dataclasses.fields() to avoid manual field-name lists that
        can drift out of sync with the actual field definitions.
        """
        issues: list[str] = []
        for prefix, sub in (
            ("db", self.db),
            ("ml", self.ml),
            ("services", self.services),
            ("events", self.events),
        ):
            for name in _missing_fields(sub):
                issues.append(f"{prefix}.{name}")
        return issues

    async def dispose(self) -> None:
        """Tear down all resources in reverse order.

        Note: DatabaseManager lifecycle is managed externally (passed in
        via init()), so we do not close it here. If ownership changes in
        the future, add ``self.db.dispose()`` here.
        """
        if not self._initialized:
            return
        await self.events.dispose()
        self.ml.dispose()
        self._initialized = False
        log.info("Infrastructure container disposed")

    # -----------------------------------------------------------------------
    # Backward-compatible convenience accessors
    #
    # These delegate to the sub-container properties so that existing code
    # using ``infra.ml_clients`` continues to work while the canonical path
    # is ``infra.ml.clients``.  New code should use the sub-container path.
    # -----------------------------------------------------------------------

    @property
    def database(self) -> DatabaseManager:
        return _require(self.db.database, "database")

    @property
    def uow_factory(self) -> UnitOfWorkFactory:
        return _require(self.db.uow_factory, "uow_factory")

    @property
    def config_broadcaster(self) -> PostgresConfigBroadcaster:
        return _require(self.db.config_broadcaster, "config_broadcaster")

    @property
    def vector_store_repo(self) -> QdrantVectorStoreRepository:
        return _require(self.ml.vector_store_repo, "vector_store_repo")

    @property
    def file_storage(self) -> LazyStorage:
        return _require(self.ml.file_storage, "file_storage")

    @property
    def document_parser(self) -> LangchainDocumentParser:
        return _require(self.ml.document_parser, "document_parser")

    @property
    def document_splitter(self) -> LangchainDocumentSplitter:
        return _require(self.ml.document_splitter, "document_splitter")

    @property
    def ml_clients(self) -> MLClientRegistry:
        return _require(self.ml.ml_clients, "ml_clients")

    @property
    def config_listener(self) -> PostgresConfigListener:
        return _require(self.events.config_listener, "config_listener")

    @property
    def health_probe(self) -> SystemHealthProbe:
        return _require(self.services.health_probe, "health_probe")

    @property
    def metrics_registry(self) -> PrometheusMetricsRegistry:
        return _require(self.ml.metrics_registry, "metrics_registry")

    @property
    def ollama_probe(self) -> OllamaProbe:
        return _require(self.services.ollama_probe, "ollama_probe")

    @property
    def qdrant_info(self) -> QdrantInfo:
        return _require(self.services.qdrant_info, "qdrant_info")

    @property
    def benchmark_service(self) -> BenchmarkService:
        return _require(self.ml.benchmark_service, "benchmark_service")

    @property
    def summary_updater(self) -> RollingSummaryUpdater:
        return _require(self.ml.summary_updater, "summary_updater")

    @property
    def api_key_provider(self) -> ApiKeyProvider:
        return _require(self.services.api_key_provider, "api_key_provider")

    @property
    def content_extractor(self) -> MLContentExtractor:
        return _require(self.ml.content_extractor, "content_extractor")

    @property
    def pdf_quality_assessor(self) -> MLPDFQualityAssessor:
        return _require(self.ml.pdf_quality_assessor, "pdf_quality_assessor")

    @property
    def metrics_collector(self) -> PrometheusMetricsCollector:
        return _require(self.ml.metrics_collector, "metrics_collector")

    @property
    def preview_cache(self) -> PreviewCache:
        return _require(self.ml.preview_cache, "preview_cache")

    @property
    def outbox_dispatcher(self) -> OutboxDispatcher:
        return _require(self.events.outbox_dispatcher, "outbox_dispatcher")

    @property
    def outbox_listener(self) -> PostgresOutboxListener:
        return _require(self.events.outbox_listener, "outbox_listener")
