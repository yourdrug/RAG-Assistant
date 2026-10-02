"""Service factories — creates services that need infrastructure dependencies.

These factory methods were previously in InfrastructureContainer but violated
the dependency direction (infrastructure → application).  Now they live here,
in the composition layer, which is allowed to depend on both infrastructure
and application.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from composition.utils import _require

if TYPE_CHECKING:
    from composition.infrastructure import InfrastructureContainer
    from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


def create_sweep_engine(infra: InfrastructureContainer):
    """Bind sweep execution to the already initialized benchmark and ML services."""
    from infrastructure.benchmark.sweep_engine import SweepEngine
    from infrastructure.benchmark.sweep_settings import LiveSweepSettings

    return SweepEngine(
        uow_factory=_require(infra.db.uow_factory, "uow_factory"),
        benchmark_service=_require(infra.ml.benchmark_service, "benchmark_service"),
        ml_clients=_require(infra.ml.ml_clients, "ml_clients"),
        runtime=LiveSweepSettings(),
    )


def create_ingestion_service(
    infra: InfrastructureContainer,
    uow_factory: UnitOfWorkFactory | None = None,
):
    """Create an IngestionService using infrastructure singletons.

    When *uow_factory* is ``None``, use the initialized application UoW.
    Ingestion always requires a UoW because database sync and registry writes
    are part of its correctness contract.
    """
    from application.services.act_versioning_service import ActVersioningService
    from application.services.batch_ingestion import BatchIngestionWorkflow
    from application.services.document_loader import S3DocumentLoader
    from application.services.ingestion_orchestrator import IngestionService
    from application.services.ingestion_registry import IngestionRegistry
    from application.services.ingestion_sync import DocumentSyncService
    from application.services.ingestion_targets import S3IngestionTargets, S3UploadService
    from application.services.single_file_ingestion import SingleFileIngestionWorkflow
    from application.services.sparse_index_builder import SparseIndexBuilder
    from infrastructure.bm25.sparse_index_admin import S3SparseIndexAdmin
    from infrastructure.ingestion.document_parser_adapter import (
        IngestionDocumentParser,
        IngestionDocumentSplitter,
    )
    from infrastructure.ml.config.settings_adapters import LiveIngestionSettings

    uow = _require(
        uow_factory if uow_factory is not None else infra.db.uow_factory,
        "uow_factory",
    )

    act_versioning = None
    if infra.domain_registry is not None and infra.domain_settings is not None:
        # Unified versioning: CLI ingestion creates act versions the same way
        # the API upload path does (TZ section 8.4)
        act_versioning = ActVersioningService(
            uow_factory=uow,
            settings=infra.domain_settings,
            cache_invalidator=infra.services.cache_invalidator,
        )

    file_storage = _require(infra.ml.file_storage, "file_storage")

    registry = IngestionRegistry(uow, file_storage)
    sync = DocumentSyncService(uow, act_versioning, infra.domain_registry, infra.domain_settings)
    ingestion_settings = LiveIngestionSettings()
    loader = S3DocumentLoader(
        file_storage,
        IngestionDocumentParser(_require(infra.ml.document_parser, "document_parser")),
        IngestionDocumentSplitter(_require(infra.ml.document_splitter, "document_splitter")),
        ingestion_settings,
        domain_registry=infra.domain_registry,
        domain_settings=infra.domain_settings,
        registry=registry,
    )
    targets = S3IngestionTargets()
    return IngestionService(
        batch_workflow=BatchIngestionWorkflow(
            _require(infra.ml.vector_store_repo, "vector_store_repo"),
            ingestion_settings,
            loader,
            registry,
            sync,
            SparseIndexBuilder(
                ingestion_settings,
                S3SparseIndexAdmin(file_storage=file_storage),
            ),
        ),
        single_file_workflow=SingleFileIngestionWorkflow(file_storage, loader, registry, sync),
        registry=registry,
        targets=targets,
        uploads=S3UploadService(file_storage, targets),
    )


def create_document_processor(
    infra: InfrastructureContainer,
    uow_factory: UnitOfWorkFactory | None = None,
):
    """Create a DocumentProcessor using infrastructure singletons."""
    from application.services.act_versioning_service import ActVersioningService
    from application.services.document_processor import DocumentProcessor
    from config import settings
    from infrastructure.ml.guardrails.text_quality_adapter import TextQualityAssessorAdapter

    uow = _require(
        uow_factory if uow_factory is not None else infra.db.uow_factory,
        "uow_factory",
    )

    act_versioning = None
    if infra.domain_registry is not None:
        act_versioning = ActVersioningService(
            uow_factory=uow,
            settings=infra.domain_settings,  # type: ignore[arg-type]
            cache_invalidator=infra.services.cache_invalidator,
        )

    return DocumentProcessor(
        uow_factory=uow,
        vector_store_repo=_require(infra.ml.vector_store_repo, "vector_store_repo"),
        file_storage=_require(infra.ml.file_storage, "file_storage"),
        document_parser=_require(infra.ml.document_parser, "document_parser"),
        document_splitter=_require(infra.ml.document_splitter, "document_splitter"),
        content_extractor=_require(infra.ml.content_extractor, "content_extractor"),
        pdf_quality_assessor=_require(infra.ml.pdf_quality_assessor, "pdf_quality_assessor"),
        text_quality_assessor=TextQualityAssessorAdapter(),
        metrics=_require(infra.ml.metrics_collector, "metrics_collector"),
        domain_marker_threshold=settings.document_domain_marker_threshold,
        domain_registry=infra.domain_registry,
        domain_settings=infra.domain_settings,
        act_versioning_service=act_versioning,
    )


def create_ingest_app_service(
    infra: InfrastructureContainer,
    uow_factory: UnitOfWorkFactory | None = None,
):
    """Create an IngestAppService using infrastructure singletons.

    Composes the IngestionService (from ``create_ingestion_service``) with
    the application-layer orchestrator.  Worker tasks call this instead of
    importing ``IngestAppService`` directly, keeping the infrastructure →
    application dependency out of the worker layer.
    """
    from application.services.ingest_service import IngestAppService

    uow = _require(
        uow_factory if uow_factory is not None else infra.db.uow_factory,
        "uow_factory",
    )
    ingestion_svc = create_ingestion_service(infra, uow_factory=uow)
    return IngestAppService(uow_factory=uow, ingestion_service=ingestion_svc)
