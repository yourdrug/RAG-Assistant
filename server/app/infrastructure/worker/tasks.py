"""Arq task bodies; shared execution policies live in worker.runtime."""

from __future__ import annotations

import logging
from typing import Any

from composition.service_providers import (
    create_document_processor,
    create_ingestion_service,
)
from application.documents.dto import ProcessDocumentCommand
from domain.knowledge.enums import DocumentVisibility
from application.benchmark import load_benchmark_questions
from domain.benchmark.run import BenchmarkRun
from application.benchmark import BenchmarkResultService
from application.benchmark import BenchmarkExportService
from infrastructure.benchmark.corpus import benchmark_corpus, corpus_metadata
from infrastructure.benchmark.checkpoint import fingerprint
from infrastructure.worker.runtime import job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document processing tasks
# ---------------------------------------------------------------------------


@job(per_principal=True)
async def process_document(
    ctx: dict[str, Any],
    *,
    document_id: int,
    storage_key: str,
    filename: str,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    replace_id: int | None,
    job_id: int,
    doc_domain: str | None = None,
    principal_id: int | None = None,
) -> None:
    """Process an uploaded document (parse → split → vectorize → store)."""
    infra = ctx["container"].infrastructure
    uow_factory = infra.db.uow_factory
    processor = create_document_processor(infra, uow_factory=uow_factory)

    async with uow_factory.create() as uow:
        doc = await uow.documents.get_by_id(document_id)

    if doc is None:
        logger.info(
            "Worker: document %d already deleted — skipping processing",
            document_id,
        )
        return

    logger.info(
        "Worker: background upload started: %s (doc %d, job %d)",
        filename,
        document_id,
        job_id,
    )
    await processor.process(ProcessDocumentCommand(
        document_id=document_id,
        storage_key=storage_key,
        original_filename=filename,
        visibility=DocumentVisibility.validate(visibility),
        owner_id=owner_id,
        group_id=group_id,
        replace_id=replace_id,
        doc_domain=doc_domain,
    ))
    logger.info(
        "Worker: background upload completed: %s (doc %d, job %d)",
        filename,
        document_id,
        job_id,
    )


async def ingest_document_scope(
    ctx: dict[str, Any],
    path: str,
    *,
    single: bool,
    domain: str,
    visibility: str,
    group_id: int | None,
    client_id: int | None,
    reset: bool = False,
    force: bool = False,
) -> None:
    """Dispatch ingestion with the same service and validated visibility scope."""
    infra = ctx["container"].infrastructure
    service = create_ingestion_service(infra, uow_factory=infra.db.uow_factory)
    vis = DocumentVisibility.validate(visibility)

    if single:
        await service.run_single(
            path, force=force, domain=domain, visibility=vis, group_id=group_id, client_id=client_id
        )
    else:
        await service.run_full(
            path, reset=reset, domain=domain, visibility=vis, group_id=group_id, client_id=client_id
        )


@job()
async def run_full_ingest(
    ctx: dict[str, Any], *, resolved_dir: str, reset: bool, domain: str, job_id: int,
    visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
    group_id: int | None = None, client_id: int | None = None,
) -> None:
    await ingest_document_scope(ctx, resolved_dir, single=False, reset=reset, domain=domain,
                                visibility=visibility, group_id=group_id, client_id=client_id)


@job()
async def run_single_ingest(
    ctx: dict[str, Any], *, resolved: str, domain: str, job_id: int,
    visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
    group_id: int | None = None, client_id: int | None = None, force: bool = False,
) -> None:
    await ingest_document_scope(ctx, resolved, single=True, force=force, domain=domain,
                                visibility=visibility, group_id=group_id, client_id=client_id)


@job()
async def run_benchmark(
    ctx: dict[str, Any],
    *,
    dataset: str,
    out_dir: str | None = None,
    top_k: int,
    judge_model: str,
    job_id: int,
) -> None:
    """Run RAG quality benchmark (non-blocking async version)."""
    uow_factory = ctx["container"].infrastructure.db.uow_factory

    service = BenchmarkResultService(uow_factory)
    existing = await service.get_job_result(job_id)

    if existing is None:
        registry = ctx["container"].infrastructure.ml.ml_clients
        async with benchmark_corpus(registry) as corpus:
            questions = await load_benchmark_questions(uow_factory, dataset)
            summary = await ctx["container"].infrastructure.ml.benchmark_service.run(
                questions=questions,
                top_k=top_k,
                judge_model=judge_model,
            )
            existing = await service.record_result(
                BenchmarkRun(
                    job_id=job_id,
                    config_json={
                        "top_k": top_k,
                        "judge_model": judge_model,
                        "evaluation_mode": "full",
                        "dataset_hash": fingerprint(questions),
                        **corpus_metadata(corpus),
                    },
                    summary_metrics={
                        key: value for key, value in summary.items() if key not in {"results", "status"}
                    },
                    per_question_results=summary["results"],
                    duration_sec=summary["total_time_sec"],
                    dataset=dataset,
                    llm_evaluated=True,
                )
            )

    if existing.id is not None:
        exports = BenchmarkExportService(uow_factory, ctx["container"].infrastructure.ml.file_storage)

        for format in ("json", "csv"):
            try:
                await exports.save_report(existing.id, format)
            except Exception:
                # Committed evaluation remains successful even if metadata/storage is unavailable.
                logger.exception("Benchmark artifact deferred: run=%s format=%s", existing.id, format)
