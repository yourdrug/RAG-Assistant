"""Composition Root — Dependency Injection Container.

Wires all services together. Every dependency flows inward: presentation → application → domain.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

from application.services.ingest_service import IngestAppService
from application.uow import UnitOfWork
from config import settings
from infrastructure.repositories.qdrant_vector_store_repository import QdrantVectorStoreRepository
from infrastructure.services.benchmark_service import BenchmarkService
from infrastructure.services.ingestion_service import IngestionService
from infrastructure.storage import get_storage
from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Shared infrastructure instances (singletons)
# ---------------------------------------------------------------------------

_vector_store_repo = QdrantVectorStoreRepository()
_file_storage = get_storage()
_uow_factory = UnitOfWorkFactory()

_ingestion_service = IngestionService(
    vector_store_repo=_vector_store_repo,
    file_storage=_file_storage,
    uow_factory=_uow_factory,
)


# ---------------------------------------------------------------------------
# Unit of Work
# ---------------------------------------------------------------------------


def get_uow() -> Generator[UnitOfWork, None, None]:
    """FastAPI dependency — yields a Unit of Work with transaction boundary."""
    with _uow_factory.create() as uow:
        yield uow


def get_uow_factory() -> UnitOfWorkFactory:
    """Return the shared UnitOfWorkFactory for background tasks."""
    return _uow_factory


# ---------------------------------------------------------------------------
# Application Services
# ---------------------------------------------------------------------------


def create_ingest_service() -> IngestAppService:
    """Create ingest application service."""
    return IngestAppService(
        uow_factory=_uow_factory,
        ingestion_service=_ingestion_service,
    )


def create_ingestion_service() -> IngestionService:
    """Create ingestion infrastructure service."""
    return _ingestion_service


# ---------------------------------------------------------------------------
# Benchmark Service
# ---------------------------------------------------------------------------


def create_benchmark_service():
    from application.use_cases.benchmark.run_benchmark import RunBenchmark

    return RunBenchmark(benchmark_service=BenchmarkService(), settings=settings)
