"""Assemble application dependencies for Arq task execution."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from application.benchmark import BenchmarkExportService, BenchmarkResultService
from application.benchmark.jobs import BenchmarkJobService
from application.benchmark.sweep_jobs import SweepJobService
from application.documents.delete_job import DocumentDeletionJob
from composition.service_providers import (
    create_document_processor,
    create_ingestion_service,
    create_sweep_engine,
)
from infrastructure.benchmark.corpus import benchmark_scope
from infrastructure.worker.sweep import publish_sweep_event


if TYPE_CHECKING:
    from composition.application import ApplicationContainer
    from composition.infrastructure import InfrastructureContainer


def worker_dependencies(infra: InfrastructureContainer, application: ApplicationContainer) -> dict[str, Any]:
    """Build lightweight services after worker configuration bootstrap."""
    if (
        application.document_command_service is None
        or infra.services.cache_invalidator is None
        or infra.services.action_logger is None
    ):
        message = "Document worker dependencies must be initialized"
        raise RuntimeError(message)

    exports = BenchmarkExportService(infra.db.uow, infra.ml.file_storage)

    if infra.ml.benchmark_service is None:
        message = "Benchmark service must be initialized"
        raise RuntimeError(message)

    results = BenchmarkResultService(infra.db.uow)
    return {
        "uow_factory": infra.db.uow,
        "file_storage": infra.ml.file_storage,
        "document_deletion_job": DocumentDeletionJob(
            infra.db.uow,
            application.document_command_service,
            infra.services.cache_invalidator,
            infra.services.action_logger,
        ),
        "benchmark_job": BenchmarkJobService(
            infra.db.uow,
            infra.ml.benchmark_service,
            results,
            exports,
            partial(benchmark_scope, infra.ml.ml_clients),
        ),
        "document_processor": create_document_processor(infra),
        "ingestion_service": create_ingestion_service(infra),
        "sweep_engine": create_sweep_engine(infra),
        "sweep_jobs": SweepJobService(infra.db.uow, publish_sweep_event, exports),
        "benchmark_results": results,
        "benchmark_exports": exports,
    }
