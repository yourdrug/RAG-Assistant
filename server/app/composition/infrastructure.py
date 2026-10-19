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
from functools import partial

from config import settings
from infrastructure.budget.ledger import PostgresRedisBudget
from application.budget import BudgetAdminService
from application.benchmark import BenchmarkService
from infrastructure.budget.runtime import budget_runtime
from infrastructure.budget.provider_catalog import OpenRouterBillingCatalog
from infrastructure.corpus.control import PostgresCorpusControl
from infrastructure.rate_limit.limiter import DynamicRateLimiter
from infrastructure.redis.redis_client import redis_client

from infrastructure.bm25.persistence import load_configured_bm25_snapshot
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.storage import LazyStorage
from typing import TYPE_CHECKING, Any

from composition.utils import _missing_fields, _require
from domain.events.config_events import ConfigParameterChanged
from infrastructure.events.in_process_event_bus import event_bus
from infrastructure.domain_profile import register_all_profiles
from domain.domain_profile.registry import DomainProfileRegistry
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.initialization import seed_domain_config_defaults, seed_config_defaults
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from infrastructure.benchmark.judge import close_judge_clients
from infrastructure.ml.config.config_subscribers import (
    audit_log_config_change,
    invalidate_paddle_ocr_cache,
    invalidate_pii_detector_cache,
)

if TYPE_CHECKING:
    from application.ports.action_logger import ActionLoggerPort
    from application.identity import ApiKeyProviderPort
    from application.benchmark import BenchmarkHistoryPort
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.admin import ConfigMaskerPort
    from application.ports.event_bus import EventBus
    from application.ports.http_metrics import HttpMetricsPort
    from application.ports.idempotency_store import IdempotencyStorePort
    from application.ports.job_enqueuer import JobEnqueuerPort
    from application.admin import LogBufferPort
    from application.documents import PreviewStrategyFactoryPort
    from application.documents import PreviewCache
    from infrastructure.database.database import DatabaseManager
    from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster
    from infrastructure.events.postgres_config_listener import PostgresConfigListener
    from infrastructure.health.system_health_probe import OllamaProbe, QdrantInfo, SystemHealthProbe
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
        from infrastructure.benchmark.runner_adapter import AsyncBenchmarkRunner
        from application.documents import PreviewCache

        self.file_storage = LazyStorage()
        self.ml_clients = MLClientRegistry(
            bm25_loader=partial(load_configured_bm25_snapshot, self.file_storage)
        )
        self.vector_store_repo = QdrantVectorStoreRepository(ml_clients=self.ml_clients)
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
        cache_invalidator: CacheInvalidatorPort | None = None,
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
            cache_invalidator=cache_invalidator,
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
    rate_limit: DynamicRateLimiter | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)
    corpus_control: PostgresCorpusControl | None = field(default=None, repr=False)
    budget_admin: BudgetAdminService | None = field(default=None, repr=False)
    configuration: SettingsProviderConfiguration | None = field(default=None, repr=False)
    _config_handlers: list = field(default_factory=list, repr=False)

    def init(self, database_manager: DatabaseManager) -> None:
        """Create all infrastructure-layer objects.

        Order matters: domain registry → DB → ML → Services → Events
        (ML parsing/splitting is domain-aware and consumes the registry).
        """
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
            cache_invalidator=self.services.cache_invalidator,
        )
        self.init_rate_limit()
        session_factory = partial(database_manager.get_session, master=True)
        budget_ledger = PostgresRedisBudget(session_factory, redis_client.async_redis)
        billing_catalog = OpenRouterBillingCatalog()
        budget_runtime.configure(budget_ledger, session_factory, billing_catalog)
        self.budget_admin = BudgetAdminService(budget_ledger, billing_catalog)
        self.configuration = SettingsProviderConfiguration(
            domain_settings=self.domain_settings,
            registry=self.domain_registry,
            bus=event_bus,
            ml_clients=self.ml.clients,
        )
        if self.events.config_listener is None:
            raise RuntimeError("Config listener must be initialized")
        self.events.config_listener.configuration = self.configuration
        self.corpus_control = PostgresCorpusControl(
            session_factory, self.ml.clients, self.events.config_listener
        )
        self.subscribe_config_events()

        self._initialized = True
        log.info("Infrastructure container initialized")

    async def bootstrap_configuration(self, role: str = "cli") -> None:
        """Load required configuration for API, worker and ML CLI."""
        if self.configuration is None or self.events.config_listener is None:
            raise RuntimeError("Configuration must be initialized before bootstrap")
        await seed_config_defaults(self.db.uow)
        await seed_domain_config_defaults(self.db.uow, self.domain_registry)
        if (
            self.budget_admin is not None
            and budget_runtime.service is not None
            and budget_runtime.service.ledger is self.budget_admin.repository
        ):
            budget_runtime.bootstrapped = True

        await self.events.config_listener.synchronize(require_complete=True)
        if self.corpus_control is None:
            raise RuntimeError("Corpus control must be initialized")
        await self.corpus_control.start(role)

    def subscribe_config_events(self) -> None:
        if self._config_handlers:
            return
        for handler in (
            self.request_configuration_sync,
            self.invalidate_config_clients,
            invalidate_paddle_ocr_cache,
            invalidate_pii_detector_cache,
            audit_log_config_change,
        ):
            event_bus.subscribe(ConfigParameterChanged, handler)
            self._config_handlers.append(handler)

    def unsubscribe_config_events(self) -> None:
        for handler in self._config_handlers:
            event_bus.unsubscribe(ConfigParameterChanged, handler)
        self._config_handlers.clear()

    def request_configuration_sync(self, event: ConfigParameterChanged) -> None:
        if self.configuration is not None and not self.configuration.applying:
            if self.events.config_listener is None:
                raise RuntimeError("Config listener must be initialized")
            self.events.config_listener.schedule_sync()

    def invalidate_config_clients(self, event: ConfigParameterChanged) -> None:
        # Only the coordinator's validated snapshot can invalidate resources.
        if self.configuration is None or not self.configuration.applying:
            return
        ml = self.ml.clients
        if self.configuration.switching_clients:
            if event.key == "hybrid_enabled":
                ml.invalidate_bm25()
            return
        if event.key in {
            "llm_provider",
            "llm_model",
            "llm_temperature",
            "llm_top_p",
            "llm_num_ctx_narrow",
            "llm_num_predict_narrow",
            "llm_num_ctx_broad",
            "llm_num_predict_broad",
            "openrouter_model",
        }:
            ml.invalidate_llm()
        if event.key == "hybrid_enabled":
            ml.invalidate_bm25()
        if event.key in {"ml_provider", "deepinfra_embed_model"}:
            ml.invalidate_embeddings()
        if event.key in {"ml_provider", "deepinfra_rerank_model"}:
            ml.invalidate_reranker()

    def init_rate_limit(self) -> None:
        """Keep an adapter even when disabled so DB settings can enable it later."""
        self.rate_limit = DynamicRateLimiter(redis_client.async_redis, settings)
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
        try:
            if (
                self.budget_admin is not None
                and budget_runtime.service is not None
                and budget_runtime.service.ledger is self.budget_admin.repository
            ):
                budget_runtime.bootstrapped = False
            if self.corpus_control is not None:
                await self.corpus_control.close()
            await close_judge_clients()
        finally:
            try:
                if self.rate_limit is not None:
                    await self.rate_limit.aclose()
                    self.rate_limit = None
            finally:
                try:
                    await self.events.dispose()
                finally:
                    try:
                        if self.ml.ml_clients is not None:
                            await self.ml.ml_clients.close()
                            self.ml.ml_clients = None
                    finally:
                        try:
                            if self.ml.file_storage is not None:
                                await self.ml.file_storage.aclose()
                        finally:
                            self.ml.dispose()
                            self.unsubscribe_config_events()
                            self._initialized = False
        log.info("Infrastructure container disposed")
