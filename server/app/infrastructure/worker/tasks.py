"""Arq task functions — wrappers around existing background processing logic.

Each task function mirrors the corresponding ``_process_document_in_background``
/ ``_tracked_ingest`` / ``_run`` from the route modules, but is designed to
run in a separate worker process via Arq.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import settings
from domain.value_objects.sweep_status import BenchmarkSweepStatus

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
    infra = ctx["container"].infrastructure
    uow_factory = infra.uow_factory

    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_running(job_id)

        processor = infra.create_document_processor(uow_factory=uow_factory)

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
    from application.services.ingest_service import IngestAppService

    infra = ctx["container"].infrastructure
    uow_factory = infra.uow_factory

    ingestion_svc = infra.create_ingestion_service(uow_factory=uow_factory)
    service = IngestAppService(uow_factory=uow_factory, ingestion_service=ingestion_svc)

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
    from application.services.ingest_service import IngestAppService

    infra = ctx["container"].infrastructure
    uow_factory = infra.uow_factory

    ingestion_svc = infra.create_ingestion_service(uow_factory=uow_factory)
    service = IngestAppService(uow_factory=uow_factory, ingestion_service=ingestion_svc)

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
    uow_factory = ctx["container"].infrastructure.uow_factory

    from infrastructure.services.benchmark_service import BenchmarkService

    service = BenchmarkService()

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
        logger.exception("Worker: benchmark failed (job %d)", job_id)
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_failed(job_id, str(e)[:500])
        except Exception:
            logger.exception("Worker: failed to mark job %d as failed", job_id)


async def run_sweep(
    ctx: dict[str, Any],
    *,
    sweep_id: int,
    job_id: int,
) -> None:
    """Run a parameter sweep as a background job."""
    import json

    from domain.entities.benchmark_run import BenchmarkRun

    uow_factory = ctx["container"].infrastructure.uow_factory

    from infrastructure.ml.sweep_engine import SweepEngine
    from infrastructure.services.benchmark_service import BenchmarkService

    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_running(job_id)
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)
            if sweep is None:
                logger.error("Sweep %d not found", sweep_id)
                return
            await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.RUNNING.value)

        # Build progress callback that publishes to Redis
        async def _publish_progress(evaluated: int, total: int, latest: dict | None) -> None:
            try:
                from arq import create_pool
                from arq.connections import RedisSettings

                redis_settings = RedisSettings.from_dsn(settings.redis_url)
                pool = await create_pool(redis_settings)
                try:
                    message = json.dumps(
                        {
                            "evaluated": evaluated,
                            "total": total,
                            "latest": latest,
                        },
                        default=str,
                    )
                    await pool.publish(f"sweep:{sweep_id}", message)
                finally:
                    await pool.close()
            except Exception:
                logger.debug("Failed to publish sweep progress to Redis", exc_info=True)

        engine = SweepEngine(
            uow_factory=uow_factory,
            benchmark_service=BenchmarkService(),
        )

        results = await engine.run_sweep(
            sweep=sweep,
            judge_model=settings.llm_model,
            progress_callback=lambda ev, tot, res: asyncio.create_task(_publish_progress(ev, tot, res)),
        )

        # Save best run to DB
        best_run_id = None
        if results:
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
                )
                run_entity = await uow.benchmark_runs.create(run_entity)
                best_run_id = run_entity.id
                await uow.benchmark_sweeps.set_best_run(sweep_id, best_run_id)
                await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.DONE.value)

        # Publish final done event
        try:
            from arq import create_pool
            from arq.connections import RedisSettings

            redis_settings = RedisSettings.from_dsn(settings.redis_url)
            pool = await create_pool(redis_settings)
            try:
                await pool.publish(
                    f"sweep:{sweep_id}",
                    json.dumps({"done": True, "best_run_id": best_run_id, "total_results": len(results)}),
                )
            finally:
                await pool.close()
        except Exception:
            logger.debug("Failed to publish sweep done event", exc_info=True)

        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_done(job_id)

        logger.info("Sweep %d completed: %d results, best_run_id=%s", sweep_id, len(results), best_run_id)

    except Exception as e:
        logger.exception("Worker: sweep failed (sweep=%d, job=%d)", sweep_id, job_id)
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.FAILED.value)
                await uow.background_jobs.mark_failed(job_id, str(e)[:500])
        except Exception:
            logger.exception("Worker: failed to mark sweep/job as failed")
