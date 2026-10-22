"""Assemble application dependencies for Arq task execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from application.benchmark import BenchmarkExportService, BenchmarkResultService
from application.benchmark.sweep_jobs import SweepJobService
from composition.service_providers import (
    create_document_processor,
    create_ingestion_service,
    create_sweep_engine,
)
from infrastructure.worker.sweep import publish_sweep_event


if TYPE_CHECKING:
    from composition.infrastructure import InfrastructureContainer


def worker_dependencies(infra: InfrastructureContainer) -> dict[str, Any]:
    """Build lightweight services after worker configuration bootstrap."""
    exports = BenchmarkExportService(infra.db.uow, infra.ml.file_storage)
    return {
        "document_processor": create_document_processor(infra),
        "ingestion_service": create_ingestion_service(infra),
        "sweep_engine": create_sweep_engine(infra),
        "sweep_jobs": SweepJobService(infra.db.uow, publish_sweep_event, exports),
        "benchmark_results": BenchmarkResultService(infra.db.uow),
        "benchmark_exports": exports,
    }
