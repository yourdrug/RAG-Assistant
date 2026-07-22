"""Composition Root — Dependency Injection Container.

Wires all use cases, repositories, and services together.
Every dependency flows inward: presentation → application → domain.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

from application.services.ingest_service import IngestAppService
from application.uow import UnitOfWork
from application.use_cases.ingest.get_registry import GetIngestRegistry
from application.use_cases.ingest.ingest_single_file import IngestSingleFile
from application.use_cases.ingest.run_ingestion import RunIngestion
from config import settings
from infrastructure.repositories.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.services.benchmark_service import BenchmarkService
from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Unit of Work
# ---------------------------------------------------------------------------

_uow_factory = UnitOfWorkFactory()


def get_uow() -> Generator[UnitOfWork, None, None]:
    """FastAPI dependency — yields a Unit of Work with transaction boundary.

    Usage in routes:
        def my_route(uow: UnitOfWork = Depends(get_uow)):
            user = uow.users.get_by_id(1)
            # Transaction auto-commits on clean exit
    """
    with _uow_factory.create() as uow:
        yield uow


# ---------------------------------------------------------------------------
# Ingest Service
# ---------------------------------------------------------------------------


def _create_ingest_service_with_session(db):
    from infrastructure.registry import load_registry, save_registry
    from infrastructure.services.ingestion_service import IngestionService

    doc_repo = SQLAlchemyDocumentRepository(db)
    ingestion = IngestionService(document_repo=doc_repo)

    class PathResolver:
        def resolve_docs_dir(self, docs_dir: str) -> str:
            return ingestion.resolve_docs_dir(docs_dir)

        def resolve_ingest_target(self, file_path: str) -> str:
            return ingestion.resolve_ingest_target(file_path)

        def force_reindex(self, filename: str) -> None:
            ingestion.force_reindex(filename)

    class RegistryAdapter:
        def load(self) -> dict:
            return load_registry(settings.data_dir)

        def save(self, registry: dict) -> None:
            save_registry(settings.data_dir, registry)

    class IngestionAdapter:
        def run_full_ingestion(self, docs_dir: str, reset: bool = False, prefix: str | None = None) -> None:
            ingestion.run_full_ingestion(docs_dir, reset=reset, prefix=prefix)

        def run_single_file(self, file_path: str) -> None:
            ingestion.run_single_file(file_path)

    return IngestAppService(
        run_ingestion=RunIngestion(
            registry_repo=RegistryAdapter(),
            ingestion_service=IngestionAdapter(),
        ),
        ingest_single_file=IngestSingleFile(
            registry_repo=RegistryAdapter(),
            ingestion_service=IngestionAdapter(),
        ),
        get_registry=GetIngestRegistry(registry_repo=RegistryAdapter()),
        path_resolver=PathResolver(),
    )


def create_ingest_service() -> IngestAppService:
    """Create ingest service with a managed session lifecycle.

    The session is created here and passed to the service.
    The service should NOT close the session — that's the caller's responsibility.
    """
    from infrastructure.database.engine import SessionLocal

    db = SessionLocal()
    return _create_ingest_service_with_session(db)


def create_ingestion_service():
    """Create ingestion service with a managed session lifecycle."""
    from infrastructure.database.engine import SessionLocal
    from infrastructure.services.ingestion_service import IngestionService

    db = SessionLocal()
    doc_repo = SQLAlchemyDocumentRepository(db)
    return IngestionService(document_repo=doc_repo)


# ---------------------------------------------------------------------------
# Benchmark Service
# ---------------------------------------------------------------------------


def create_benchmark_service():
    from application.use_cases.benchmark.run_benchmark import RunBenchmark

    return RunBenchmark(benchmark_service=BenchmarkService(), settings=settings)
