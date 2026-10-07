"""Worker sweep tasks — parameter sweep orchestration for Arq background jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from config import settings
from domain.entities.benchmark_run import BenchmarkRun
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from composition.service_providers import create_sweep_engine
from infrastructure.benchmark.sweep_engine import SweepCancelled
from infrastructure.benchmark.checkpoint_cleanup import cleanup_sweep_checkpoints

from infrastructure.worker.outcomes import IncompleteEvaluation

logger = logging.getLogger("default")


async def publish_sweep_event(sweep_id: int, message: dict) -> None:
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


async def handle_sweep_cancelled(uow_factory, sweep_id: int) -> None:
    logger.info("Sweep %d cancelled — stopping early", sweep_id)
    async with uow_factory.create(master=True) as uow:
        await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.CANCELLED.value)
    await asyncio.to_thread(cleanup_sweep_checkpoints, settings.data_dir, sweep_id, successful=False)
    await publish_sweep_event(sweep_id, {"done": True, "cancelled": True})


async def save_sweep_results(uow_factory, sweep, sweep_id: int, results: list[dict]) -> int | None:
    """Persist every config, including partial results; return the eligible winner."""
    best_run_id = None
    for result in results:
        run_id = await save_sweep_run(uow_factory, sweep, sweep_id, result)
        if best_run_id is None and result.get("composite_score") is not None:
            best_run_id = run_id
    async with uow_factory.create(master=True) as uow:
        if best_run_id is not None:
            await uow.benchmark_sweeps.set_best_run(sweep_id, best_run_id)
        # Completion is set only after checking coverage and cancellation.
    return best_run_id


async def save_sweep_run(uow_factory, sweep, sweep_id: int, best: dict) -> int | None:
    config = best.get("config", {})
    full = best.get("full_metrics", {})
    metrics = {
        "hit_rate": full.get("hit_rate") if full else best.get("avg_hit_rate"),
        "mrr": full.get("avg_mrr") if full else best.get("avg_mrr"),
        "composite": best.get("composite_score"),
        "faithfulness": best.get("full_metrics", {}).get("avg_faithfulness"),
        "relevancy": full.get("avg_relevancy"),
        "correctness": full.get("avg_correctness"),
        "evaluation_mode": sweep.evaluation_mode,
        "evaluated_config_count": best.get("evaluated_config_count", 0),
        "scored_config_count": best.get("scored_config_count", 0),
        "search_config_count": best.get("search_config_count", 0),
        **{
            key: full.get(key)
            for key in (
                "total_questions",
                "judge_evaluated_count",
                "judge_error_count",
                "total_input_tokens",
                "total_output_tokens",
                "total_judge_input_tokens",
                "total_judge_output_tokens",
                "total_judge_calls",
                "usage_scope",
            )
        },
    }
    # Keep evidence averages and coverage counts so a composite score can be
    # explained without reopening the original report files.
    diagnostics = full or best
    metrics.update(
        {
            key: value
            for key, value in diagnostics.items()
            if key.startswith(("avg_fragment_", "avg_context_", "avg_retrieval_", "avg_source_"))
            or key.endswith(("_evaluated_count", "_expected_count"))
            or key.startswith("retrieval_")
        }
    )
    async with uow_factory.create(master=True) as uow:
        run_entity = BenchmarkRun(
            sweep_id=sweep_id,
            config_json=config,
            summary_metrics=metrics,
            dataset=sweep.dataset,
            llm_evaluated=best.get("llm_evaluated", False),
            per_question_results=best.get("full_metrics", {}).get("results"),
        )
        run_entity = await uow.benchmark_runs.save_for_sweep(run_entity)
    return run_entity.id


async def run_cancellable_sweep(engine, sweep, progress_callback, is_cancelled) -> list[dict]:
    task = asyncio.create_task(
        engine.run_sweep(
            sweep=sweep,
            judge_model=sweep.judge_model or settings.benchmark_judge_model or settings.llm_model,
            progress_callback=progress_callback,
            should_cancel=is_cancelled,
        )
    )
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.5)
            if await is_cancelled():
                raise SweepCancelled("Sweep cancelled during evaluation")
        return await task
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def run_sweep_task(
    ctx: dict[str, Any],
    *,
    sweep_id: int,
    job_id: int,
) -> None:
    """Run a parameter sweep as a background job (cooperatively cancellable)."""
    from infrastructure.worker.tasks import run_tracked_job

    uow_factory = ctx["container"].infrastructure.db.uow_factory

    async def is_cancelled() -> bool:
        return await sweep_job_cancelled(uow_factory, sweep_id, job_id)

    async def run_action() -> None:
        sweep = await start_sweep_job(uow_factory, sweep_id, job_id)
        if sweep is None:
            return

        async def progress_callback(ev: int, tot: int, res: dict | None) -> None:
            await record_sweep_progress(uow_factory, sweep, sweep_id, ev, tot, res)

        try:
            engine = create_sweep_engine(ctx["container"].infrastructure)
            results = await run_cancellable_sweep(engine, sweep, progress_callback, is_cancelled)
            best_run_id = await save_sweep_results(uow_factory, sweep, sweep_id, results)
            validate_sweep_coverage(results)
        except SweepCancelled:
            await handle_sweep_cancelled(uow_factory, sweep_id)
            return
        except asyncio.CancelledError:
            await record_sweep_failure(uow_factory, sweep_id, "Task cancelled (timeout or worker shutdown)")
            raise
        except Exception as exc:
            cancelled = await record_sweep_failure(uow_factory, sweep_id, str(exc))
            if cancelled:
                return
            raise

        await complete_sweep(uow_factory, sweep_id, results, best_run_id)

    await run_tracked_job(
        uow_factory,
        job_id,
        run_action,
        description=f"sweep {sweep_id}",
        job_try=ctx.get("job_try", 1),
        max_tries=1,
    )


def validate_sweep_coverage(results: list[dict]) -> None:
    evaluated = [r["full_metrics"] for r in results if r.get("llm_evaluated")]
    error_count = sum(r.get("judge_error_count", 0) for r in evaluated)
    incomplete = sum(r.get("evaluation_complete") is False for r in results)
    if error_count or incomplete:
        total = sum(r.get("total_questions", 0) for r in evaluated)
        complete = sum(r.get("judge_evaluated_count", 0) for r in evaluated)
        # All configurations have run. Retain checkpoints and expose Resume
        # using the existing failed-sweep lifecycle, even with a valid winner.
        raise IncompleteEvaluation(
            f"Оценено {complete} из {total}, ошибок {error_count}. "
            f"Неполных конфигураций: {incomplete}. Resume выполнит недостающие оценки."
        )


async def complete_sweep(uow_factory, sweep_id: int, results: list[dict], best_run_id: int | None) -> None:
    async with uow_factory.create(master=True) as uow:
        current = await uow.benchmark_sweeps.get_by_id(sweep_id, for_update=True)
        if current is not None and current.status == BenchmarkSweepStatus.CANCELLED.value:
            await publish_sweep_event(sweep_id, {"done": True, "cancelled": True})
            return
        await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.DONE.value)
    await asyncio.to_thread(cleanup_sweep_checkpoints, settings.data_dir, sweep_id, successful=True)
    await publish_sweep_event(
        sweep_id,
        {"done": True, "best_run_id": best_run_id, "total_results": len(results)},
    )
    logger.info("Sweep %d completed: %d results, best_run_id=%s", sweep_id, len(results), best_run_id)


def inactive_sweep_job(sweep, job_id: int) -> bool:
    return sweep.status != BenchmarkSweepStatus.PENDING.value or (
        sweep.job_id is not None and sweep.job_id != job_id
    )


async def sweep_job_cancelled(uow_factory, sweep_id: int, job_id: int) -> bool:
    async with uow_factory.create(master=True) as uow:
        sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)
        return sweep is not None and (
            sweep.status == BenchmarkSweepStatus.CANCELLED.value
            or (sweep.job_id is not None and sweep.job_id != job_id)
        )


async def record_sweep_progress(uow_factory, sweep, sweep_id, ev, tot, res) -> None:
    async with uow_factory.create(master=True) as uow:
        await uow.benchmark_sweeps.update_progress(sweep_id, ev, tot)
    if res and res.get("phase") == "full_evaluation" and "config" in res:
        await save_sweep_run(uow_factory, sweep, sweep_id, res)
    await publish_sweep_event(sweep_id, {"evaluated": ev, "total": tot, "latest": res})


async def start_sweep_job(uow_factory, sweep_id: int, job_id: int):
    async with uow_factory.create(master=True) as uow:
        sweep = await uow.benchmark_sweeps.get_by_id(sweep_id, for_update=True)
        if sweep is None:
            raise ValueError(f"Sweep {sweep_id} not found")
        if inactive_sweep_job(sweep, job_id):
            logger.info("Ignoring inactive or superseded sweep job %d", job_id)
            return
        await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.RUNNING.value)
        await uow.benchmark_sweeps.update_progress(sweep_id, 0, 0)
    return sweep


async def record_sweep_failure(uow_factory, sweep_id: int, message: str) -> bool:
    async with uow_factory.create(master=True) as uow:
        current = await uow.benchmark_sweeps.get_by_id(sweep_id, for_update=True)
        cancelled = current is not None and current.status == BenchmarkSweepStatus.CANCELLED.value
        if not cancelled:
            await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.FAILED.value)
    await asyncio.to_thread(cleanup_sweep_checkpoints, settings.data_dir, sweep_id, successful=False)
    event = {"done": True, "cancelled": True} if cancelled else {"done": True, "error": message}
    await publish_sweep_event(sweep_id, event)
    return cancelled
