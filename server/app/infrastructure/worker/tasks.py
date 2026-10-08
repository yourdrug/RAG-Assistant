"""Arq task functions — wrappers around existing background processing logic.

Each task function mirrors the corresponding ``_process_document_in_background``
/ ``_tracked_ingest`` / ``_run`` from the route modules, but is designed to
run in a separate worker process via Arq.

Job state guarantees:
- Every long task runs inside ``run_tracked_job``: RUNNING on start, DONE on
  success, FAILED on exception, and FAILED on ``CancelledError`` (arq timeout
  or worker shutdown) — documents never stay PROCESSING forever.
- A heartbeat task refreshes ``background_jobs.heartbeat_at`` every minute so
  the orphan reaper can distinguish a live long job from a dead one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from arq import Retry
from composition.service_providers import (
    create_document_processor,
    create_ingest_app_service,
)
from domain.value_objects.visibility import DocumentVisibility
from application.services.benchmark_dataset import load_benchmark_questions
from infrastructure.benchmark.runner import run_benchmark_async
from infrastructure.worker.admission import one_document_per_principal
from infrastructure.worker.outcomes import IncompleteEvaluation, job_error_message

# Re-export cron tasks for backward compatibility (Arq worker config imports from here)
from infrastructure.worker.cron import (  # noqa: F401
    cron_bm25_rebuild,
    cron_job_cleanup,
    cron_recover_orphaned_jobs,
    cron_recover_stuck_processing,
)

logger = logging.getLogger("default")

_HEARTBEAT_INTERVAL_SEC = 60
_MAX_JOB_TRIES = 3


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _job_heartbeat(uow_factory, job_id: int, interval_sec: int = _HEARTBEAT_INTERVAL_SEC):
    """Periodically touch ``heartbeat_at`` while the job body runs."""

    async def _beat() -> None:
        while True:
            await asyncio.sleep(interval_sec)
            try:
                async with uow_factory.create(master=True) as uow:
                    await uow.background_jobs.touch_heartbeat(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Heartbeat update failed for job %d", job_id)

    task = asyncio.create_task(_beat(), name=f"job-heartbeat-{job_id}")
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _mark_job_failed_safe(uow_factory, job_id: int, error: str) -> None:
    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_failed(job_id, error[:500])
    except Exception:
        logger.exception("Worker: failed to mark job %d as failed", job_id)


async def run_tracked_job(
    uow_factory,
    job_id: int,
    action: Callable[[], Awaitable[None]],
    *,
    description: str,
    job_try: int = 1,
    max_tries: int = _MAX_JOB_TRIES,
) -> None:
    """Run a job body with full job-state bookkeeping (C-4)."""
    async with _job_heartbeat(uow_factory, job_id, interval_sec=_HEARTBEAT_INTERVAL_SEC):
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_running(job_id)
        except Exception:
            logger.exception("Worker: failed to mark job %d as running", job_id)

        try:
            await action()
        except asyncio.CancelledError:
            logger.warning("Worker: job %d cancelled (%s) — persisting failed state", job_id, description)
            await _mark_job_failed_safe(uow_factory, job_id, "Task cancelled (timeout or worker shutdown)")
            raise
        except IncompleteEvaluation as exc:
            logger.warning("Worker: job %d incomplete (%s): %s", job_id, description, exc)
            await _mark_job_failed_safe(uow_factory, job_id, str(exc))
        except Exception as e:
            logger.exception("Worker: job %d failed (%s)", job_id, description)
            if job_try < max_tries:
                defer_seconds = min(60, 5 * (2 ** (job_try - 1)))
                logger.warning(
                    "Worker: retrying job %d (attempt %d/%d) in %ds",
                    job_id,
                    job_try,
                    max_tries,
                    defer_seconds,
                )
                raise Retry(defer=defer_seconds) from e
            await _mark_job_failed_safe(uow_factory, job_id, job_error_message(e))
        else:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_done(job_id)


# ---------------------------------------------------------------------------
# Document processing tasks
# ---------------------------------------------------------------------------


@one_document_per_principal
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

    async def _action() -> None:
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

    await run_tracked_job(
        uow_factory,
        job_id,
        _action,
        description=f"process document {document_id}",
        job_try=ctx.get("job_try", 1),
    )


async def run_full_ingest(
    ctx: dict[str, Any],
    *,
    resolved_dir: str,
    reset: bool,
    domain: str,
    job_id: int,
    visibility: str = "internal_public",
    group_id: int | None = None,
    client_id: int | None = None,
) -> None:
    """Full document ingestion from a directory."""
    infra = ctx["container"].infrastructure
    uow_factory = infra.db.uow_factory

    service = create_ingest_app_service(infra, uow_factory=uow_factory)

    vis = DocumentVisibility.validate(visibility)

    async def _action() -> None:
        await service.run_full(
            resolved_dir, reset, domain=domain, visibility=vis, group_id=group_id, client_id=client_id
        )

    await run_tracked_job(
        uow_factory,
        job_id,
        _action,
        description=f"full ingest {resolved_dir}",
        job_try=ctx.get("job_try", 1),
    )


async def run_single_ingest(
    ctx: dict[str, Any],
    *,
    resolved: str,
    domain: str,
    job_id: int,
    visibility: str = "internal_public",
    group_id: int | None = None,
    client_id: int | None = None,
) -> None:
    """Ingest a single file."""
    infra = ctx["container"].infrastructure
    uow_factory = infra.db.uow_factory

    service = create_ingest_app_service(infra, uow_factory=uow_factory)

    vis = DocumentVisibility.validate(visibility)

    async def _action() -> None:
        await service.run_single(
            resolved, domain=domain, visibility=vis, group_id=group_id, client_id=client_id
        )

    await run_tracked_job(
        uow_factory,
        job_id,
        _action,
        description=f"single ingest {resolved}",
        job_try=ctx.get("job_try", 1),
    )


async def run_benchmark(
    ctx: dict[str, Any],
    *,
    dataset: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
    job_id: int,
) -> None:
    """Run RAG quality benchmark (non-blocking async version)."""
    uow_factory = ctx["container"].infrastructure.db.uow_factory

    async def evaluate_benchmark() -> None:
        questions = await load_benchmark_questions(uow_factory, dataset)
        results = await run_benchmark_async(
            questions=questions,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
            export_files=False,
        )
        from application.services.benchmark_orchestrator import compute_summary_from_results
        from domain.entities.benchmark_run import BenchmarkRun

        summary = compute_summary_from_results(results)
        async with uow_factory.create(master=True) as uow:
            await uow.benchmark_runs.create(
                BenchmarkRun(
                    config_json={"top_k": top_k, "judge_model": judge_model},
                    summary_metrics={key: value for key, value in summary.items() if key != "results"},
                    per_question_results=summary["results"],
                    dataset=dataset,
                    llm_evaluated=True,
                )
            )

    await run_tracked_job(
        uow_factory,
        job_id,
        evaluate_benchmark,
        description="benchmark run",
        job_try=ctx.get("job_try", 1),
    )
