"""Service factories — creates services that need infrastructure dependencies.

These factory methods were previously in InfrastructureContainer but violated
the dependency direction (infrastructure → application).  Now they live here,
in the composition layer, which is allowed to depend on both infrastructure
and application.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import TYPE_CHECKING, Any

from application.act_versioning import ActVersioningService
from application.benchmark.sweep import SweepBenchmarkDependencies, SweepEngine
from application.benchmark.sweep_evaluation import SweepFullEvaluator
from application.documents import DocumentProcessor
from application.documents.dto import DocumentProcessingDependencies
from application.ingestion import (
    BatchIngestionWorkflow,
    DocumentSyncService,
    IngestionRegistry,
    IngestionService,
    S3DocumentLoader,
    S3IngestionTargets,
    S3UploadService,
    SingleFileIngestionWorkflow,
)
from application.rag.service import RagService as ApplicationRagService
from composition.utils import require
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.benchmark.corpus import benchmark_corpus, corpus_metadata
from infrastructure.benchmark.sweep.data import SweepDataSource
from infrastructure.benchmark.sweep.scoring import score_stage_candidates
from infrastructure.benchmark.sweep.settings import LiveSweepSettings
from infrastructure.benchmark.sweep.strategies import default_strategy_factories
from infrastructure.ingestion.document_parser_adapter import (
    IngestionDocumentParser,
    IngestionDocumentSplitter,
)
from infrastructure.ml.config.settings_adapters import LiveIngestionSettings
from infrastructure.ml.guardrails.text_quality_adapter import TextQualityAssessorAdapter
from infrastructure.ml.rag.rag_steps import RagOperations


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from application.benchmark import BenchmarkCheckpoints
    from application.benchmark.sweep import SweepBenchmarkPort
    from application.benchmark.sweep_ports import SweepSearchStrategy, SweepSettingsPort
    from application.chat import PIIRedactorPort
    from application.documents import DocumentAccessPort
    from application.ports.ml_clients import MLClientPort
    from application.uow.factory import UnitOfWorkFactory
    from composition.infrastructure import InfrastructureContainer
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.rag.ports import ChunkRetrievalRepository

log = logging.getLogger("default")


def build_sweep_engine(
    uow_factory: UnitOfWorkFactory,
    benchmark_service: SweepBenchmarkPort | None = None,
    ml_clients: MLClientPort | None = None,
    *,
    runtime: SweepSettingsPort | None = None,
    strategies: dict[str, Callable[[dict[str, Any]], SweepSearchStrategy]] | None = None,
    checkpoint_factory: Callable[[int], BenchmarkCheckpoints] | None = None,
) -> SweepEngine:
    @asynccontextmanager
    async def corpus() -> AsyncIterator[dict[str, Any]]:
        async with benchmark_corpus(ml_clients) as snapshot:
            yield corpus_metadata(snapshot)

    runtime = runtime if runtime is not None else LiveSweepSettings()
    return SweepEngine(
        data=SweepDataSource(uow_factory, ml_clients),
        benchmark=SweepBenchmarkDependencies(benchmark_service, checkpoint_factory),
        runtime=runtime,
        full_evaluator=SweepFullEvaluator(runtime),
        strategies=strategies if strategies is not None else default_strategy_factories(),
        scoring=score_stage_candidates,
        corpus=corpus,
    )


def create_sweep_engine(infra: InfrastructureContainer) -> SweepEngine:
    return build_sweep_engine(
        uow_factory=require(infra.db.uow_factory, "uow_factory"),
        benchmark_service=require(infra.ml.benchmark_service, "benchmark_service"),
        ml_clients=require(infra.ml.ml_clients, "ml_clients"),
        checkpoint_factory=lambda sweep_id: DatabaseBenchmarkCheckpoints(
            require(infra.db.uow_factory, "uow_factory"), sweep_id
        ),
    )


def create_ingestion_service(
    infra: InfrastructureContainer,
    uow_factory: UnitOfWorkFactory | None = None,
) -> IngestionService:
    """Create an IngestionService using infrastructure singletons.

    When *uow_factory* is ``None``, use the initialized application UoW.
    Ingestion always requires a UoW because database sync and registry writes
    are part of its correctness contract.
    """
    uow = require(
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

    file_storage = require(infra.ml.file_storage, "file_storage")

    registry = IngestionRegistry(uow, file_storage)
    sync = DocumentSyncService(uow, act_versioning, infra.domain_registry, infra.domain_settings)
    ingestion_settings = LiveIngestionSettings()
    loader = S3DocumentLoader(
        file_storage,
        IngestionDocumentParser(require(infra.ml.document_parser, "document_parser")),
        IngestionDocumentSplitter(require(infra.ml.document_splitter, "document_splitter")),
        ingestion_settings,
        domain_registry=infra.domain_registry,
        domain_settings=infra.domain_settings,
        registry=registry,
    )
    targets = S3IngestionTargets()
    return IngestionService(
        batch_workflow=BatchIngestionWorkflow(
            infra.ml.vector_index_writer,
            ingestion_settings,
            loader,
            registry,
            sync=sync,
        ),
        single_file_workflow=SingleFileIngestionWorkflow(
            file_storage,
            loader,
            registry,
            sync,
        ),
        registry=registry,
        targets=targets,
        uploads=S3UploadService(file_storage, targets),
    )


def create_document_processor(
    infra: InfrastructureContainer,
    uow_factory: UnitOfWorkFactory | None = None,
) -> DocumentProcessor:
    """Create a DocumentProcessor using infrastructure singletons."""
    uow = require(
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
        file_storage=require(infra.ml.file_storage, "file_storage"),
        dependencies=DocumentProcessingDependencies(
            document_parser=IngestionDocumentParser(require(infra.ml.document_parser, "document_parser")),
            document_splitter=IngestionDocumentSplitter(
                require(infra.ml.document_splitter, "document_splitter")
            ),
            content_extractor=require(infra.ml.content_extractor, "content_extractor"),
            pdf_quality_assessor=require(infra.ml.pdf_quality_assessor, "pdf_quality_assessor"),
            text_quality_assessor=TextQualityAssessorAdapter(),
            metrics=require(infra.ml.metrics_collector, "metrics_collector"),
        ),
        domain_registry=infra.domain_registry,
        domain_settings=infra.domain_settings,
        act_versioning_service=act_versioning,
    )


# Compatibility for integrations using the former application facade factory.
create_ingest_app_service = create_ingestion_service


def create_rag_service(
    ml_clients: MLClientPort,
    chunk_search: ChunkRetrievalRepository | None = None,
    domain_keys: tuple[str, ...] | None = None,
    pii_redactor: PIIRedactorPort | None = None,
    *,
    document_access: DocumentAccessPort,
    domain_registry: DomainProfileRegistry | None = None,
) -> ApplicationRagService:
    operations = RagOperations(
        ml_clients,
        chunk_search,
        domain_keys,
        pii_redactor,
        document_access=document_access,
        domain_registry=domain_registry,
    )
    return ApplicationRagService(
        preparation=operations,
        safety=operations,
        cache=operations,
        retrieval=operations,
        evidence=operations,
        context=operations,
        generation=operations,
        finalization=operations,
        corpus=operations.corpus,
    )
