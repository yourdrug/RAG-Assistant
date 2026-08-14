"""Arq task functions — wrappers around existing background processing logic.

Each task function mirrors the corresponding ``_process_document_in_background``
/ ``_tracked_ingest`` / ``_run`` from the route modules, but is designed to
run in a separate worker process via Arq.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("default")


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
) -> None:
    """Process an uploaded document (parse → split → vectorize → store)."""
    from application.services.document_processor import DocumentProcessor  # nested to avoid circular import
    from presentation.api.dependencies import (  # nested to avoid circular import
        get_document_parser,
        get_document_splitter,
        get_file_storage,
        get_uow_factory,
        get_vector_store_repo,
    )

    uow_factory = get_uow_factory()

    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_running(job_id)

        processor = DocumentProcessor(
            uow_factory=uow_factory,
            vector_store_repo=get_vector_store_repo(),
            file_storage=get_file_storage(),
            document_parser=get_document_parser(),
            document_splitter=get_document_splitter(),
        )

        logger.info(
            "Worker: background upload started: %s (doc %d, job %d)",
            filename,
            document_id,
            job_id,
        )
        await processor.process(
            document_id=document_id,
            storage_key=storage_key,
            original_filename=filename,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
            replace_id=replace_id,
            doc_domain=doc_domain,
        )
        logger.info(
            "Worker: background upload completed: %s (doc %d, job %d)",
            filename,
            document_id,
            job_id,
        )
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_done(job_id)
    except Exception as e:
        logger.exception(
            "Worker: background document processing failed for %s (doc %d, job %d)",
            filename,
            document_id,
            job_id,
        )
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_failed(job_id, str(e)[:500])
        except Exception:
            logger.exception("Worker: failed to mark job %d as failed", job_id)


async def run_full_ingest(
    ctx: dict[str, Any],
    *,
    resolved_dir: str,
    reset: bool,
    domain: str,
    job_id: int,
) -> None:
    """Full document ingestion from a directory."""
    from presentation.api.dependencies import get_uow_factory  # nested to avoid circular import
    from presentation.api.routes.ingest import create_ingest_service  # nested to avoid circular import

    uow_factory = get_uow_factory()
    service = create_ingest_service()

    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_running(job_id)
        await service.run_full(resolved_dir, reset, domain=domain)
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_done(job_id)
    except Exception as e:
        logger.exception("Worker: background ingest failed (job %d)", job_id)
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_failed(job_id, str(e)[:500])
        except Exception:
            logger.exception("Worker: failed to mark job %d as failed", job_id)


async def run_single_ingest(
    ctx: dict[str, Any],
    *,
    resolved: str,
    domain: str,
    job_id: int,
) -> None:
    """Ingest a single file."""
    from presentation.api.dependencies import get_uow_factory  # nested to avoid circular import
    from presentation.api.routes.ingest import create_ingest_service  # nested to avoid circular import

    uow_factory = get_uow_factory()
    service = create_ingest_service()

    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_running(job_id)
        await service.run_single(resolved, domain=domain)
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_done(job_id)
    except Exception as e:
        logger.exception("Worker: background single-file ingest failed (job %d)", job_id)
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_failed(job_id, str(e)[:500])
        except Exception:
            logger.exception("Worker: failed to mark job %d as failed", job_id)


async def run_benchmark(
    ctx: dict[str, Any],
    *,
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
    job_id: int,
) -> None:
    """Run RAG quality benchmark."""
    from presentation.api.dependencies import (  # nested to avoid circular import
        create_benchmark_service,
        get_uow_factory,
    )

    uow_factory = get_uow_factory()
    service = create_benchmark_service()

    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_running(job_id)
        service.run(
            questions_path=questions_path,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
        )
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_done(job_id)
    except Exception as e:
        logger.error("Worker: benchmark failed: %s", e)
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_failed(job_id, str(e)[:500])
        except Exception:
            logger.exception("Worker: failed to mark job %d as failed", job_id)
