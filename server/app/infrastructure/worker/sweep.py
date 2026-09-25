"""Worker sweep tasks — parameter sweep orchestration for Arq background jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from config import settings
from domain.entities.benchmark_run import BenchmarkRun
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from infrastructure.benchmark.sweep_engine import SweepEngine, SweepCancelled

logger = logging.getLogger("default")


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


async def run_sweep_task(
    ctx: dict[str, Any],
    *,
    sweep_id: int,
    job_id: int,
) -> None:
    """Run a parameter sweep as a background job (cooperatively cancellable)."""
    from infrastructure.worker.tasks import _run_tracked_job

    uow_factory = ctx["container"].infrastructure.db.uow_factory

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
            benchmark_service=ctx["container"].infrastructure.ml.benchmark_service,
            ml_clients=ctx["container"].infrastructure.ml.ml_clients,
            rag_service=ctx["container"].application.rag_service,
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
            async with uow_factory.create(master=True) as uow:
                await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.FAILED.value)
            raise

        best_run_id = await _save_sweep_results(uow_factory, sweep, sweep_id, results)
        await _publish_sweep_event(
            sweep_id,
            {"done": True, "best_run_id": best_run_id, "total_results": len(results)},
        )
        logger.info("Sweep %d completed: %d results, best_run_id=%s", sweep_id, len(results), best_run_id)

    await _run_tracked_job(uow_factory, job_id, _action, description=f"sweep {sweep_id}")
