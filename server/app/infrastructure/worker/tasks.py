"""Arq task functions — wrappers around existing background processing logic.

Each task function mirrors the corresponding ``_process_document_in_background``
/ ``_tracked_ingest`` / ``_run`` from the route modules, but is designed to
run in a separate worker process via Arq.

Job state guarantees:
- Every long task runs inside ``_run_tracked_job``: RUNNING on start, DONE on
  success, FAILED on exception, and FAILED on ``CancelledError`` (arq timeout
  or worker shutdown) — documents never stay PROCESSING forever.
- A heartbeat task refreshes ``background_jobs.heartbeat_at`` every minute so
  the orphan reaper can distinguish a live long job from a dead one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from application.services.ingest_service import IngestAppService
from composition.service_providers import create_document_processor, create_ingestion_service
from config import settings
from domain.entities.benchmark_run import BenchmarkRun
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.benchmark.sweep_engine import SweepEngine, SweepCancelled
from infrastructure.benchmark.benchmark import run_benchmark_async
from infrastructure.services.benchmark_service import BenchmarkService

logger = logging.getLogger("default")

_HEARTBEAT_INTERVAL_SEC = 60


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


async def _run_tracked_job(
    uow_factory,
    job_id: int,
    action: Callable[[], Awaitable[None]],
    *,
    description: str,
) -> None:
    """Run a job body with full job-state bookkeeping (C-4).

    - marks the job running,
    - keeps a heartbeat alive while it runs,
    - marks done/failed on completion,
    -     on ``CancelledError`` (arq timeout / worker shutdown) marks the job
      failed and re-raises so arq's own accounting stays correct.
    """
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
        except Exception as e:
            logger.exception("Worker: job %d failed (%s)", job_id, description)
            await _mark_job_failed_safe(uow_factory, job_id, str(e))
        else:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_done(job_id)


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
    infra = ctx["container"].infrastructure
    uow_factory = infra.uow_factory
    processor = create_document_processor(infra, uow_factory=uow_factory)

    async def _action() -> None:
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

    await _run_tracked_job(uow_factory, job_id, _action, description=f"process document {document_id}")


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
    uow_factory = infra.uow_factory

    ingestion_svc = create_ingestion_service(infra, uow_factory=uow_factory)
    service = IngestAppService(uow_factory=uow_factory, ingestion_service=ingestion_svc)

    vis = DocumentVisibility.validate(visibility)

    async def _action() -> None:
        await service.run_full(
            resolved_dir, reset, domain=domain, visibility=vis, group_id=group_id, client_id=client_id
        )

    await _run_tracked_job(uow_factory, job_id, _action, description=f"full ingest {resolved_dir}")


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
    uow_factory = infra.uow_factory

    ingestion_svc = create_ingestion_service(infra, uow_factory=uow_factory)
    service = IngestAppService(uow_factory=uow_factory, ingestion_service=ingestion_svc)

    vis = DocumentVisibility.validate(visibility)

    async def _action() -> None:
        await service.run_single(
            resolved, domain=domain, visibility=vis, group_id=group_id, client_id=client_id
        )

    await _run_tracked_job(uow_factory, job_id, _action, description=f"single ingest {resolved}")


async def run_benchmark(
    ctx: dict[str, Any],
    *,
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
    job_id: int,
) -> None:
    """Run RAG quality benchmark (non-blocking async version)."""
    uow_factory = ctx["container"].infrastructure.uow_factory

    async def _action() -> None:
        await run_benchmark_async(
            questions_path=questions_path,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
        )

    await _run_tracked_job(uow_factory, job_id, _action, description="benchmark run")


async def _publish_sweep_event(sweep_id: int, message: dict) -> None:
    """Publish a sweep progress/done event to the Redis pub/sub channel."""
    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        redis_settings = RedisSettings.from_dsn(settings.redis_url)
        pool = await create_pool(redis_settings)
        try:
            await pool.publish(f"sweep:{sweep_id}", json.dumps(message, default=str))
        finally:
            await pool.close()
    except Exception:
        logger.debug("Failed to publish sweep event to Redis", exc_info=True)


async def run_sweep(
    ctx: dict[str, Any],
    *,
    sweep_id: int,
    job_id: int,
) -> None:
    """Run a parameter sweep as a background job (cooperatively cancellable)."""
    uow_factory = ctx["container"].infrastructure.uow_factory

    async def _is_cancelled() -> bool:
        async with uow_factory.create() as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)
            return sweep is not None and sweep.status == BenchmarkSweepStatus.CANCELLED.value

    async def _action() -> None:
        async with uow_factory.create(master=True) as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)
            if sweep is None:
                logger.error("Sweep %d not found", sweep_id)
                return
            await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.RUNNING.value)

        def _progress_callback(ev: int, tot: int, res: dict | None) -> None:
            message = {"evaluated": ev, "total": tot, "latest": res}
            asyncio.create_task(_publish_sweep_event(sweep_id, message))

        engine = SweepEngine(
            uow_factory=uow_factory,
            benchmark_service=BenchmarkService(),
            ml_clients=ctx["container"].infrastructure.ml_clients,
        )

        try:
            results = await engine.run_sweep(
                sweep=sweep,
                judge_model=settings.llm_model,
                progress_callback=_progress_callback,
                should_cancel=_is_cancelled,
            )
        except SweepCancelled:
            await _handle_sweep_cancelled(uow_factory, sweep_id)
            return
        except Exception:
            # Keep the sweep status consistent with the failed job (C-4)
            async with uow_factory.create(master=True) as uow:
                await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.FAILED.value)
            raise

        # Save best run to DB
        best_run_id = await _save_sweep_results(uow_factory, sweep, sweep_id, results)
        await _publish_sweep_event(
            sweep_id,
            {"done": True, "best_run_id": best_run_id, "total_results": len(results)},
        )
        logger.info("Sweep %d completed: %d results, best_run_id=%s", sweep_id, len(results), best_run_id)

    await _run_tracked_job(uow_factory, job_id, _action, description=f"sweep {sweep_id}")


async def _handle_sweep_cancelled(uow_factory, sweep_id: int) -> None:
    logger.info("Sweep %d cancelled — stopping early", sweep_id)
    async with uow_factory.create(master=True) as uow:
        await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.CANCELLED.value)
    await _publish_sweep_event(sweep_id, {"done": True, "cancelled": True})


async def _save_sweep_results(uow_factory, sweep, sweep_id: int, results: list[dict]) -> int | None:
    """Persist the best run and mark the sweep done. Returns the best run id."""
    best_run_id = None
    if not results:
        return best_run_id
    best = results[0]
    config = best.get("config", {})
    metrics = {
        "hit_rate": best.get("avg_hit_rate"),
        "mrr": best.get("avg_mrr"),
        "composite": best.get("composite_score"),
        "faithfulness": best.get("full_metrics", {}).get("avg_faithfulness"),
        "relevancy": best.get("full_metrics", {}).get("avg_relevancy"),
    }
    async with uow_factory.create(master=True) as uow:
        run_entity = BenchmarkRun(
            sweep_id=sweep_id,
            config_json=config,
            summary_metrics=metrics,
            dataset=sweep.dataset,
            llm_evaluated=best.get("llm_evaluated", False),
            per_question_results=best.get("full_metrics", {}).get("results"),
        )
        run_entity = await uow.benchmark_runs.create(run_entity)
        best_run_id = run_entity.id
        await uow.benchmark_sweeps.set_best_run(sweep_id, best_run_id)
        await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.DONE.value)
    return best_run_id


# ---------------------------------------------------------------------------
# Cron tasks — periodic maintenance, run by Arq worker on schedule
# ---------------------------------------------------------------------------


async def cron_job_cleanup(ctx: dict[str, Any]) -> None:
    """Delete old background job records (runs every hour)."""
    uow_factory = ctx["container"].infrastructure.uow_factory
    async with uow_factory.create(master=True) as uow:
        deleted = await uow.background_jobs.delete_old(days=settings.job_cleanup_days)
        if deleted:
            logger.info("Cron: cleaned up %d old background jobs", deleted)


async def cron_recover_orphaned_jobs(ctx: dict[str, Any]) -> None:
    """Fail jobs whose heartbeat expired and pending jobs never picked up (every 15 min)."""
    uow_factory = ctx["container"].infrastructure.uow_factory
    async with uow_factory.create(master=True) as uow:
        orphaned_ids = await uow.background_jobs.recover_orphaned(timeout_minutes=15)
        stale_pending = await uow.background_jobs.fail_stale_pending(timeout_minutes=30)
    if orphaned_ids:
        logger.warning("Cron: recovered %d orphaned jobs: %s", len(orphaned_ids), orphaned_ids)
    for job in stale_pending:
        logger.warning("Cron: failed stale pending job %d (%s)", job.id, job.job_type)
    await _fail_documents_of_dead_jobs(uow_factory, stale_pending)


async def _fail_documents_of_dead_jobs(uow_factory, dead_jobs) -> None:
    """Mark documents failed whose processing job died before pickup."""
    doc_ids = [
        job.related_id for job in dead_jobs if job.job_type == "document_processing" and job.related_id
    ]
    if not doc_ids:
        return
    async with uow_factory.create(master=True) as uow:
        for doc_id in doc_ids:
            doc = await uow.documents.get_by_id(doc_id)
            if doc is not None and doc.status.value in ("pending", "processing"):
                await uow.documents.update_status(
                    doc_id,
                    "failed",
                    error="Processing job was never picked up — enqueue lost or worker unavailable",
                )
                logger.warning("Cron: marked document %d failed (dead job)", doc_id)


async def cron_recover_stuck_processing(ctx: dict[str, Any]) -> None:
    """Fail documents stuck in 'processing' with no active background job (every 15 min).

    A document is in 'processing' only while its job runs; a PROCESSING row
    without a pending/running job means the worker died mid-task (e.g. OOM
    or SIGKILL — nothing could write the failure state).
    """
    uow_factory = ctx["container"].infrastructure.uow_factory
    async with uow_factory.create(master=True) as uow:
        stuck_ids = await uow.documents.mark_stuck_processing_failed()
    for doc_id in stuck_ids:
        logger.warning("Cron: marked stuck PROCESSING document %d as failed", doc_id)


async def cron_bm25_rebuild(ctx: dict[str, Any]) -> None:
    """Rebuild BM25 index from scratch (runs daily at 3:00 AM UTC)."""
    if not settings.hybrid_enabled:
        logger.debug("Cron: BM25 rebuild skipped — hybrid search disabled")
        return

    from infrastructure.bm25.bm25_invalidation import publish_bm25_invalidation
    from infrastructure.bm25.hybrid import BM25Index, save_bm25_index_to_s3
    from infrastructure.storage import get_storage

    uow_factory = ctx["container"].infrastructure.uow_factory
    t0 = time.monotonic()

    async with uow_factory.create(master=True) as uow:
        all_texts = await uow.chunks.get_all_contents()

    if not all_texts:
        logger.info("Cron: BM25 rebuild — no chunks found, skipping")
        return

    bm25_index = BM25Index(all_texts)
    storage = get_storage()
    await save_bm25_index_to_s3(bm25_index, storage)
    await publish_bm25_invalidation()

    elapsed = time.monotonic() - t0
    logger.info(
        "Cron: BM25 rebuild completed — %d chunks indexed in %.1fs",
        len(all_texts),
        elapsed,
    )
