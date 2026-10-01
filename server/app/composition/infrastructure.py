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
from typing import TYPE_CHECKING, Any

from composition.utils import _missing_fields, _require

if TYPE_CHECKING:
    from application.ports.action_logger import ActionLoggerPort
    from application.ports.api_key_provider import ApiKeyProviderPort
    from application.ports.benchmark_history import BenchmarkHistoryPort
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.ports.config_masker import ConfigMaskerPort
    from application.ports.event_bus import EventBus
    from application.ports.http_metrics import HttpMetricsPort
    from application.ports.idempotency_store import IdempotencyStorePort
    from application.ports.job_enqueuer import JobEnqueuerPort
    from application.ports.log_buffer import LogBufferPort
    from application.ports.preview_strategy_factory import PreviewStrategyFactoryPort
    from application.services.preview_cache import PreviewCache
    from infrastructure.database.database import DatabaseManager
    from domain.domain_profile.registry import DomainProfileRegistry
    from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
    from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster
    from infrastructure.events.postgres_config_listener import PostgresConfigListener
    from infrastructure.health.system_health_probe import OllamaProbe, QdrantInfo, SystemHealthProbe
    from infrastructure.ml.clients.client_registry import MLClientRegistry
    from infrastructure.ml.extraction.extraction_adapter import MLContentExtractor, MLPDFQualityAssessor
    from infrastructure.ml.langchain_document_parser import (
        LangchainDocumentParser,
        LangchainDocumentSplitter,
    )
    from infrastructure.metrics.metrics_adapter import PrometheusMetricsCollector
    from infrastructure.metrics.prometheus_adapter import PrometheusMetricsRegistry
    from infrastructure.rate_limit.limiter import PyrateRateLimiter
    from infrastructure.ml.extraction.summary_adapter import RollingSummaryUpdater
    from infrastructure.repositories.vector.qdrant_vector_store_repository import (
        QdrantVectorStoreRepository,
    )
    from application.services.benchmark_orchestrator import BenchmarkService
    from infrastructure.storage import LazyStorage
    from infrastructure.uow_factory import UnitOfWorkFactory
    from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher
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

    def init(
        self,
        uow_factory: UnitOfWorkFactory,
        domain_registry: DomainProfileRegistry | None = None,
        domain_settings: DomainSettingsAdapter | None = None,
    ) -> None:
        from infrastructure.ml.clients.client_registry import MLClientRegistry
        from infrastructure.ml.extraction.extraction_adapter import MLContentExtractor, MLPDFQualityAssessor
        from infrastructure.ml.langchain_document_parser import (
            LangchainDocumentParser,
            LangchainDocumentSplitter,
        )
        from infrastructure.metrics.metrics_adapter import PrometheusMetricsCollector
        from infrastructure.metrics.prometheus_adapter import PrometheusMetricsRegistry
        from infrastructure.ml.extraction.summary_adapter import RollingSummaryUpdater
        from infrastructure.repositories.vector.qdrant_vector_store_repository import (
            QdrantVectorStoreRepository,
        )
        from application.services.benchmark_orchestrator import BenchmarkService
        from infrastructure.benchmark.runner_adapter import AsyncBenchmarkRunner
        from infrastructure.storage import LazyStorage
        from application.services.preview_cache import PreviewCache

        self.ml_clients = MLClientRegistry()
        self.vector_store_repo = QdrantVectorStoreRepository(ml_clients=self.ml_clients)
        self.file_storage = LazyStorage()
        self.preview_cache = PreviewCache(storage=self.file_storage)
        self.document_parser = LangchainDocumentParser(
            domain_registry=domain_registry,
            domain_settings=domain_settings,
        )
        self.document_splitter = LangchainDocumentSplitter(
            domain_registry=domain_registry,
            domain_settings=domain_settings,
        )
        self.metrics_registry = PrometheusMetricsRegistry()
        self.benchmark_service = BenchmarkService(rag_service=None, runner=AsyncBenchmarkRunner())
        self.summary_updater = RollingSummaryUpdater(ml_clients=self.ml_clients)
        self.content_extractor = MLContentExtractor()
        self.pdf_quality_assessor = MLPDFQualityAssessor()
        self.metrics_collector = PrometheusMetricsCollector()

    def dispose(self) -> None:
        """Clear ML-specific caches and release resources."""
        # LazyStorage proxies the resolved backend; the cache lives on the
        # get_storage() factory (lru_cache) — reset that, not the proxy.
        from infrastructure.storage import get_storage

        get_storage.cache_clear()

        # Clear OCR/lru_cache caches — import may fail if optional deps
        # (paddleocr, surya) are not installed.
        try:
            from infrastructure.ml.ingestion import get_paddle_ocr, get_surya_predictors
        except ImportError:
            return
        try:
            get_paddle_ocr.cache_clear()
            get_surya_predictors.cache_clear()
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
        event_bus: EventBus,
        domain_settings: DomainSettingsAdapter | None = None,
    ) -> None:
        from infrastructure.events.postgres_config_listener import PostgresConfigListener
        from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher
        from infrastructure.events.postgres_outbox_listener import PostgresOutboxListener
        from config import settings as app_settings

        self.config_listener = PostgresConfigListener(
            event_bus=event_bus,
            uow_factory=uow_factory,
            domain_settings=domain_settings,
        )
        self.outbox_dispatcher = OutboxDispatcher(
            uow_factory=uow_factory,
            vector_store=vector_store_repo,
        )
        self.outbox_listener = PostgresOutboxListener(
            dispatcher=self.outbox_dispatcher,
            dsn=f"postgresql://{app_settings.db_user}:{app_settings.db_password}"
            f"@{app_settings.db_host}:{app_settings.db_port}/{app_settings.db_name}",
        )

    async def dispose(self) -> None:
        """Stop listeners and close dispatcher."""
        if self.config_listener is not None:
            await self.config_listener.stop()
        if self.outbox_listener is not None:
            await self.outbox_listener.stop()


class PreviewStrategyFactoryAdapter:
    """Instance adapter for PreviewStrategyFactory (static methods)."""

    def for_extension(self, extension: str, **kwargs):
        from infrastructure.ml.preview.factory import PreviewStrategyFactory

        return PreviewStrategyFactory.for_extension(extension, **kwargs)

    def supported_extensions(self) -> set[str]:
        from infrastructure.ml.preview.factory import PreviewStrategyFactory

        return PreviewStrategyFactory.supported_extensions()


class HttpMetricsAdapter:
    """Adapter wrapping a prometheus_client Counter behind HttpMetricsPort."""

    def __init__(self, counter: Any) -> None:
        self._counter = counter

    def inc_requests(self, *, handler: str, method: str, status: str) -> None:
        self._counter.labels(handler=handler, method=method, status=status).inc()


@dataclass
class ServiceContainer:
    """Auxiliary infrastructure services: health, admin, auth, presentation adapters."""

    health_probe: SystemHealthProbe | None = field(default=None)
    ollama_probe: OllamaProbe | None = field(default=None)
    qdrant_info: QdrantInfo | None = field(default=None)
    api_key_provider: ApiKeyProviderPort | None = field(default=None)
    action_logger: ActionLoggerPort | None = field(default=None)
    cache_invalidator: CacheInvalidatorPort | None = field(default=None)
    job_enqueuer: JobEnqueuerPort | None = field(default=None)
    config_masker: ConfigMaskerPort | None = field(default=None)
    log_buffer: LogBufferPort | None = field(default=None)
    preview_strategy_factory: PreviewStrategyFactoryPort | None = field(default=None)
    benchmark_history: BenchmarkHistoryPort | None = field(default=None)
    idempotency_store: IdempotencyStorePort | None = field(default=None)
    http_metrics: HttpMetricsPort | None = field(default=None)

    def init(self) -> None:
        from infrastructure.auth.api_key_provider import api_key_provider
        from infrastructure.health.system_health_probe import OllamaProbe, QdrantInfo, SystemHealthProbe
        from infrastructure.adapters.action_logger_adapter import ActionLoggerAdapter
        from infrastructure.adapters.cache_invalidator_adapter import CacheInvalidatorAdapter
        from infrastructure.adapters.job_enqueuer_adapter import JobEnqueuerAdapter
        from infrastructure.adapters.config_masker_adapter import ConfigMaskerAdapter
        from infrastructure.adapters.log_buffer_adapter import LogBufferAdapter
        from infrastructure.benchmark.benchmark_history_adapter import BenchmarkHistoryAdapter
        from infrastructure.rate_limit.idempotency import IdempotencyStore
        from infrastructure.redis.redis_client import redis_client

        self.health_probe = SystemHealthProbe()
        self.ollama_probe = OllamaProbe(health_probe=self.health_probe)
        self.qdrant_info = QdrantInfo(health_probe=self.health_probe)
        self.api_key_provider = api_key_provider
        self.action_logger = ActionLoggerAdapter()
        self.cache_invalidator = CacheInvalidatorAdapter()
        self.job_enqueuer = JobEnqueuerAdapter()
        self.config_masker = ConfigMaskerAdapter()
        self.log_buffer = LogBufferAdapter()
        self.preview_strategy_factory = PreviewStrategyFactoryAdapter()
        self.benchmark_history = BenchmarkHistoryAdapter()
        self.idempotency_store = IdempotencyStore(redis=redis_client.async_redis)

        from infrastructure.metrics.metrics import HTTP_REQUESTS_TOTAL

        self.http_metrics = HttpMetricsAdapter(HTTP_REQUESTS_TOTAL)

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
    def api_keys(self) -> ApiKeyProviderPort:
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
    domain_registry: DomainProfileRegistry | None = field(default=None, repr=False)
    domain_settings: DomainSettingsAdapter | None = field(default=None, repr=False)
    rate_limit: PyrateRateLimiter | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def init(self, database_manager: DatabaseManager) -> None:
        """Create all infrastructure-layer objects.

        Order matters: domain registry → DB → ML → Services → Events
        (ML parsing/splitting is domain-aware and consumes the registry).
        """
        from infrastructure.domain_profile import register_all_profiles
        from domain.domain_profile.registry import DomainProfileRegistry
        from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
        from infrastructure.events.in_process_event_bus import event_bus

        # Domain profile registry — created first: ML parsing/splitting is domain-aware
        self.domain_settings = DomainSettingsAdapter()
        self.domain_registry = DomainProfileRegistry()
        register_all_profiles(self.domain_registry, self.domain_settings)

        self.db.init(database_manager)
        self.ml.init(
            uow_factory=self.db.uow,
            domain_registry=self.domain_registry,
            domain_settings=self.domain_settings,
        )
        self.services.init()
        self.events.init(
            uow_factory=self.db.uow,
            vector_store_repo=self.ml.vector_store,
            event_bus=event_bus,
            domain_settings=self.domain_settings,
        )
        self._init_rate_limit()

        self._initialized = True
        log.info("Infrastructure container initialized")

    def _init_rate_limit(self) -> None:
        """Build the Redis-backed rate limiter (requires an initialized Redis client)."""
        from config import settings
        from infrastructure.rate_limit.limiter import PyrateRateLimiter
        from infrastructure.rate_limit.policies import build_policies

        if not settings.rate_limit_enabled:
            log.info("Rate limiting disabled (RATE_LIMIT_ENABLED=false)")
            return

        from infrastructure.redis.redis_client import redis_client

        # Raises if redis_client.init() has not run yet — lifespan calls it first.
        self.rate_limit = PyrateRateLimiter(
            redis_client.async_redis,
            build_policies(settings),
            key_prefix=settings.rate_limit_redis_prefix,
            max_buckets=settings.rate_limit_max_buckets,
            fail_open=settings.rate_limit_fail_open,
        )
        if settings.forwarded_allow_ips.strip() == "127.0.0.1":
            log.warning(
                "Rate limiting is enabled but FORWARDED_ALLOW_IPS=127.0.0.1 — behind a reverse "
                "proxy all clients share the proxy IP bucket (login limit). See server/.env.example."
            )

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
        if self.rate_limit is not None:
            await self.rate_limit.aclose()
            self.rate_limit = None
        await self.events.dispose()
        if self.ml.file_storage is not None:
            await self.ml.file_storage.aclose()
        self.ml.dispose()
        self._initialized = False
        log.info("Infrastructure container disposed")
