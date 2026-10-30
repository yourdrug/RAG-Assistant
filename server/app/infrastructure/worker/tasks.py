"""Arq task bodies; shared execution policies live in worker.runtime."""

from __future__ import annotations

import logging
from typing import Any

from application.documents.dto import ProcessDocumentCommand
from domain.knowledge.enums import DocumentVisibility
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
    processor = ctx["document_processor"]

    logger.info(
        "Worker: background upload started: %s (doc %d, job %d)",
        filename,
        document_id,
        job_id,
    )
    await processor.process(
        ProcessDocumentCommand(
            document_id=document_id,
            storage_key=storage_key,
            original_filename=filename,
            visibility=DocumentVisibility.parse(visibility),
            owner_id=owner_id,
            group_id=group_id,
            replace_id=replace_id,
            doc_domain=doc_domain,
        )
    )
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
    service = ctx["ingestion_service"]
    vis = DocumentVisibility.parse(visibility)

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
    ctx: dict[str, Any],
    *,
    resolved_dir: str,
    reset: bool,
    domain: str,
    job_id: int,
    visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
    group_id: int | None = None,
    client_id: int | None = None,
) -> None:
    await ingest_document_scope(
        ctx,
        resolved_dir,
        single=False,
        reset=reset,
        domain=domain,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
    )


@job()
async def run_single_ingest(
    ctx: dict[str, Any],
    *,
    resolved: str,
    domain: str,
    job_id: int,
    visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
    group_id: int | None = None,
    client_id: int | None = None,
    force: bool = False,
) -> None:
    await ingest_document_scope(
        ctx,
        resolved,
        single=True,
        force=force,
        domain=domain,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
    )


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
    await ctx["benchmark_job"].run(dataset=dataset, top_k=top_k, judge_model=judge_model, job_id=job_id)
