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

from application.benchmark.sweep import SweepBenchmarkDependencies, SweepEngine
from application.benchmark.sweep_evaluation import SweepFullEvaluator
from application.documents import DocumentProcessor
from application.documents.dto import DocumentProcessingDependencies
from application.documents.reindex import DocumentReindexService
from application.documents.space_processor import SpaceDocumentProcessor
from application.indexing.contextualization import ChunkContextualization
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
from application.ingestion.space_router import SpaceIngestionService
from application.rag.service import RagService as ApplicationRagService
from application.rag.space_router import SpaceRagRouter
from application.spaces import SpacePackHooks
from application.spaces.resolution import default_space_ref
from composition.utils import require
from config import ChunkingSection, process_section
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.benchmark.corpus import benchmark_corpus, corpus_metadata
from infrastructure.benchmark.space_scope import space_corpus_metadata, sweep_space_scope
from infrastructure.benchmark.sweep.data import SweepDataSource
from infrastructure.benchmark.sweep.scoring import score_stage_candidates
from infrastructure.benchmark.sweep.settings import LiveSweepSettings
from infrastructure.benchmark.sweep.strategies import default_strategy_factories
from infrastructure.domain_profile.packs import NEUTRAL_PACK_KEY
from infrastructure.ingestion.document_parser_adapter import (
    IngestionDocumentParser,
    IngestionDocumentSplitter,
)
from infrastructure.ml.cache.keys import config_snapshot
from infrastructure.ml.config.settings_adapters import LiveIngestionSettings
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, evidence_profile_scope
from infrastructure.ml.guardrails.text_quality_adapter import TextQualityAssessorAdapter
from infrastructure.ml.ingestion.llm_contextualizer import LLMChunkContextualizer, RedisNoteCache
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_steps import RagOperations
from infrastructure.redis.redis_client import redis_client


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from application.benchmark import BenchmarkCheckpoints
    from application.benchmark.sweep import SweepBenchmarkPort
    from application.benchmark.sweep_ports import SweepSearchStrategy, SweepSettingsPort
    from application.chat import PIIRedactorPort
    from application.documents import DocumentAccessPort
    from application.indexing.pack_hooks import DocumentPackHooks
    from application.ports.ml_clients import MLClientPort
    from application.uow.factory import UnitOfWorkFactory
    from composition.infrastructure import InfrastructureContainer
    from composition.pack_catalog import LoadedPackCatalog
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.rag.ports import ChunkRetrievalRepository
    from infrastructure.ml.evidence_profiles.profiles import EvidenceProfile

log = logging.getLogger("default")


def build_sweep_engine(  # noqa: PLR0913 - independent build options
    uow_factory: UnitOfWorkFactory,
    benchmark_service: SweepBenchmarkPort | None = None,
    ml_clients: MLClientPort | None = None,
    *,
    runtime: SweepSettingsPort | None = None,
    strategies: dict[str, Callable[[dict[str, Any]], SweepSearchStrategy]] | None = None,
    checkpoint_factory: Callable[[int], BenchmarkCheckpoints] | None = None,
    evidence_profile: EvidenceProfile | None = None,
    evidence_profiles: Mapping[str, EvidenceProfile] | None = None,
) -> SweepEngine:
    """Bind retrieval and full evaluation to the selected space and its pack's evidence rules."""

    async def run_profile(pack_key: str) -> EvidenceProfile:
        if evidence_profile is not None or evidence_profiles is None:
            return evidence_profile or GenericEvidenceProfile()

        return evidence_profiles[pack_key]

    @asynccontextmanager
    async def corpus(space_id: int | None = None) -> AsyncIterator[dict[str, Any]]:
        # Standalone engines may inject evidence rules without a persistent pack catalog.
        space = (
            await default_space_ref(uow_factory, space_id)
            if evidence_profiles is not None or space_id is not None
            else None
        )
        profile = await run_profile(space.pack_key if space is not None else NEUTRAL_PACK_KEY)
        with evidence_profile_scope(profile):
            async with benchmark_corpus(ml_clients) as snapshot:
                if space is None:
                    yield corpus_metadata(snapshot)
                else:
                    with sweep_space_scope(space):
                        yield space_corpus_metadata(corpus_metadata(snapshot), space)

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
    uow_factory = require(infra.db.uow_factory, "uow_factory")

    def checkpoints(sweep_id: int) -> BenchmarkCheckpoints:
        return DatabaseBenchmarkCheckpoints(uow_factory, sweep_id)

    catalog = pack_catalog(infra)
    return build_sweep_engine(
        uow_factory=uow_factory,
        benchmark_service=require(infra.ml.benchmark_service, "benchmark_service"),
        ml_clients=require(infra.ml.ml_clients, "ml_clients"),
        checkpoint_factory=checkpoints,
        evidence_profiles={key: catalog.pack(key).evidence_profile for key in catalog.pack_keys()},
    )


def create_chunk_contextualization(infra: InfrastructureContainer) -> ChunkContextualization | None:
    """Build plan 4.3 notes only when explicitly enabled (off by default)."""
    config = process_section(ChunkingSection)

    if not config.chunk_contextualization_enabled:
        return None

    clients = require(infra.ml.ml_clients, "ml_clients")
    return ChunkContextualization(
        LLMChunkContextualizer(
            clients.fast_llm,
            RedisNoteCache(redis_client.async_redis),
            concurrency=config.chunk_contextualization_concurrency,
            timeout_seconds=config.chunk_contextualization_timeout,
        ),
        infra.ml.token_counter,
    )


def pack_catalog(infra: InfrastructureContainer) -> LoadedPackCatalog:
    return require(infra.pack_catalog, "pack_catalog")


def create_pack_hooks(
    infra: InfrastructureContainer, uow: UnitOfWorkFactory, pack_key: str = NEUTRAL_PACK_KEY
) -> DocumentPackHooks | None:
    """One pack's document lifecycle hooks; ``None`` for packs without them."""
    return pack_catalog(infra).build_hooks(pack_key, uow)


def create_space_pack_hooks(infra: InfrastructureContainer) -> SpacePackHooks:
    """Hooks of each document's pack, chosen by the document's space."""
    return SpacePackHooks(pack_catalog(infra))


def pack_parsing(
    infra: InfrastructureContainer, pack_key: str
) -> tuple[IngestionDocumentParser, IngestionDocumentSplitter]:
    """Parser and splitter that know only the document profiles of one pack."""
    registry = pack_catalog(infra).domain_registry(pack_key)
    parser = LangchainDocumentParser(domain_registry=registry, domain_settings=infra.domain_settings)
    splitter = LangchainDocumentSplitter(
        token_counter=infra.ml.token_counter, domain_registry=registry, domain_settings=infra.domain_settings
    )
    return IngestionDocumentParser(parser), IngestionDocumentSplitter(splitter)


def create_ingestion_service(
    infra: InfrastructureContainer,
    uow_factory: UnitOfWorkFactory | None = None,
) -> SpaceIngestionService:
    """Ingest storage files into a space with the rules of the space's pack.

    When *uow_factory* is ``None``, use the initialized application UoW.
    Ingestion always requires a UoW because database sync and registry writes
    are part of its correctness contract.
    """
    uow = require(
        uow_factory if uow_factory is not None else infra.db.uow_factory,
        "uow_factory",
    )
    services = {
        key: create_pack_ingestion_service(infra, uow, key) for key in pack_catalog(infra).pack_keys()
    }
    return SpaceIngestionService(uow, services, services[NEUTRAL_PACK_KEY])


def create_pack_ingestion_service(
    infra: InfrastructureContainer, uow: UnitOfWorkFactory, pack_key: str
) -> IngestionService:
    file_storage = require(infra.ml.file_storage, "file_storage")
    domain_registry = pack_catalog(infra).domain_registry(pack_key)
    parser, splitter = pack_parsing(infra, pack_key)

    registry = IngestionRegistry(uow, file_storage)
    sync = DocumentSyncService(
        uow,
        # CLI ingestion records pack state the same way the API upload path does.
        create_pack_hooks(infra, uow, pack_key),
        domain_registry,
        infra.domain_settings,
        contextualization=create_chunk_contextualization(infra),
    )
    ingestion_settings = LiveIngestionSettings()
    loader = S3DocumentLoader(
        file_storage,
        parser,
        splitter,
        ingestion_settings,
        domain_registry=domain_registry,
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
) -> SpaceDocumentProcessor:
    """Process each document with the processor of its space's pack."""
    uow = require(
        uow_factory if uow_factory is not None else infra.db.uow_factory,
        "uow_factory",
    )
    processors = {
        key: create_pack_document_processor(infra, uow, key) for key in pack_catalog(infra).pack_keys()
    }
    return SpaceDocumentProcessor(uow, processors)


def create_pack_document_processor(
    infra: InfrastructureContainer, uow: UnitOfWorkFactory, pack_key: str
) -> DocumentProcessor:
    parser, splitter = pack_parsing(infra, pack_key)
    return DocumentProcessor(
        uow_factory=uow,
        file_storage=require(infra.ml.file_storage, "file_storage"),
        dependencies=DocumentProcessingDependencies(
            document_parser=parser,
            document_splitter=splitter,
            content_extractor=require(infra.ml.content_extractor, "content_extractor"),
            pdf_quality_assessor=require(infra.ml.pdf_quality_assessor, "pdf_quality_assessor"),
            text_quality_assessor=TextQualityAssessorAdapter(),
            metrics=require(infra.ml.metrics_collector, "metrics_collector"),
            chunk_contextualization=create_chunk_contextualization(infra),
        ),
        domain_registry=pack_catalog(infra).domain_registry(pack_key),
        domain_settings=infra.domain_settings,
        pack_hooks=create_pack_hooks(infra, uow, pack_key),
    )


def create_reindex_service(infra: InfrastructureContainer) -> DocumentReindexService:
    uow = require(infra.db.uow_factory, "uow_factory")
    return DocumentReindexService(uow, create_document_processor(infra, uow))


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
    evidence_profile: EvidenceProfile | None = None,
) -> ApplicationRagService:
    operations = RagOperations(
        ml_clients,
        chunk_search,
        domain_keys,
        pii_redactor,
        document_access=document_access,
        domain_registry=domain_registry,
        evidence_profile=evidence_profile,
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


def create_space_rag_router(
    infra: InfrastructureContainer,
    ml_clients: MLClientPort,
    chunk_search: ChunkRetrievalRepository,
    pii_redactor: PIIRedactorPort,
    document_access: DocumentAccessPort,
) -> SpaceRagRouter:
    """One RAG service per loaded pack: its profiles, evidence rules and answer policies."""
    catalog = pack_catalog(infra)
    services = {}

    for key in catalog.pack_keys():
        registry = catalog.domain_registry(key)
        services[key] = create_rag_service(
            ml_clients=ml_clients,
            chunk_search=chunk_search,
            domain_keys=tuple(profile.key for profile in registry.all()),
            pii_redactor=pii_redactor,
            document_access=document_access,
            domain_registry=registry,
            evidence_profile=catalog.pack(key).evidence_profile,
        )

    return SpaceRagRouter(services)


def build_config_inspector(infra: InfrastructureContainer) -> Callable[[dict[str, Any]], dict[str, Any]]:
    catalog = pack_catalog(infra)

    def inspect(saved: dict[str, Any]) -> dict[str, Any]:
        space = saved.get("space")
        pack_key = space["pack_key"] if space is not None else NEUTRAL_PACK_KEY
        profile = catalog.pack(pack_key).evidence_profile
        with evidence_profile_scope(profile):
            return config_snapshot(build_rag_settings(), space, catalog.domain_registry(pack_key))[1]

    return inspect
