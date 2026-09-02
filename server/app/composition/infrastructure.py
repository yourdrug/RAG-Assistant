"""Infrastructure sub-container — singletons for database, ML, storage, etc."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

log = logging.getLogger("default")


@dataclass
class InfrastructureContainer:
    """Infrastructure-layer singletons and factories.

    All fields are assigned in ``init()`` — not in ``__init__`` (they use
    ``field(default=None)``).  This keeps the dataclass declarative while
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
    content_extractor: MLContentExtractor | None = field(default=None)
    pdf_quality_assessor: MLPDFQualityAssessor | None = field(default=None)
    metrics_collector: PrometheusMetricsCollector | None = field(default=None)
    preview_cache: PreviewCache | None = field(default=None)

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
        from application.services.preview_cache import PreviewCache

        self.database = database_manager
        self.api_key_provider = api_key_provider
        self.config_broadcaster = PostgresConfigBroadcaster()
        self.uow_factory = UnitOfWorkFactory(
            database=database_manager,
            config_broadcaster=self.config_broadcaster,
        )
        self.ml_clients = MLClientRegistry()
        self.vector_store_repo = QdrantVectorStoreRepository(ml_clients=self.ml_clients)
        self.file_storage = LazyStorage()
        self.preview_cache = PreviewCache(storage=self.file_storage)
        self.document_parser = LangchainDocumentParser()
        self.document_splitter = LangchainDocumentSplitter()
        self.health_probe = SystemHealthProbe()
        self.metrics_registry = PrometheusMetricsRegistry()
        self.ollama_probe = OllamaProbe()
        self.qdrant_info = QdrantInfo()
        self.benchmark_service = BenchmarkService()
        self.summary_updater = RollingSummaryUpdater(ml_clients=self.ml_clients)
        self.content_extractor = MLContentExtractor()
        self.pdf_quality_assessor = MLPDFQualityAssessor()
        self.metrics_collector = PrometheusMetricsCollector()
        self.config_listener = PostgresConfigListener(
            event_bus=event_bus,
            uow_factory=self.uow_factory,
        )

    def create_ingestion_service(
        self,
        uow_factory: UnitOfWorkFactory | None = None,
    ):
        """Create an IngestionService using this container's infrastructure singletons.

        When *uow_factory* is ``None`` falls back to ``self.uow_factory``
        (the one initialised in ``init()``).  This is useful for CLI commands
        that don't need a database connection.
        """
        from infrastructure.services.ingestion_service import IngestionService

        vsr = self.vector_store_repo
        if vsr is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        fs = self.file_storage
        if fs is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        resolved_uow = uow_factory if uow_factory is not None else self.uow_factory
        if resolved_uow is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")

        return IngestionService(
            vector_store_repo=vsr,
            file_storage=fs,
            uow_factory=resolved_uow,
        )

    def create_document_processor(
        self,
        uow_factory: UnitOfWorkFactory | None = None,
    ):
        """Create a DocumentProcessor using this container's infrastructure singletons."""
        from application.services.document_processor import DocumentProcessor
        from config import settings

        vsr = self.vector_store_repo
        if vsr is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        fs = self.file_storage
        if fs is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        resolved_uow = uow_factory if uow_factory is not None else self.uow_factory
        if resolved_uow is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        parser = self.document_parser
        if parser is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        splitter = self.document_splitter
        if splitter is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        extractor = self.content_extractor
        if extractor is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        pdf_qa = self.pdf_quality_assessor
        if pdf_qa is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")
        metrics = self.metrics_collector
        if metrics is None:
            raise RuntimeError("InfrastructureContainer.init() must be called first")

        return DocumentProcessor(
            uow_factory=resolved_uow,
            vector_store_repo=vsr,
            file_storage=fs,
            document_parser=parser,
            document_splitter=splitter,
            content_extractor=extractor,
            pdf_quality_assessor=pdf_qa,
            metrics=metrics,
            domain_marker_threshold=settings.document_domain_marker_threshold,
            ml_registry=self.ml_clients,
        )

    async def dispose(self) -> None:
        """No infrastructure resources need explicit dispose — lifecycle is managed by lifespan().

        This method exists for symmetry and future use.
        """
