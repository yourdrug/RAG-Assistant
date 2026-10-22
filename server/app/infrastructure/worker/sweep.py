"""Arq sweep execution, cancellation and Redis event delivery."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from application.benchmark.sweep_jobs import validate_sweep_coverage
from application.benchmark.sweep_ports import SweepCancelled
from config import settings
from infrastructure.redis.redis_client import redis_client
from infrastructure.worker.runtime import job
from infrastructure.worker.outcomes import job_error_message

logger = logging.getLogger(__name__)


async def publish_sweep_event(sweep_id: int, message: dict) -> None:
    """Publish a sweep progress/done event to the Redis pub/sub channel."""
    try:
        await redis_client.async_redis.publish(f"sweep:{sweep_id}", json.dumps(message, default=str))
    except Exception:
        logger.debug("Failed to publish sweep event to Redis", exc_info=True)


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


@job(max_tries=1)
async def run_sweep_task(
    ctx: dict[str, Any],
    *,
    sweep_id: int,
    job_id: int,
) -> None:
    """Run a parameter sweep as a background job (cooperatively cancellable)."""
    service = ctx["sweep_jobs"]

    async def is_cancelled() -> bool:
        return await service.sweep_job_cancelled(sweep_id, job_id)

    sweep = await service.start_sweep_job(sweep_id, job_id)

    if sweep is None:
        return

    async def progress_callback(ev: int, tot: int, res: dict | None) -> None:
        await service.record_sweep_progress(sweep, sweep_id, ev, tot, res)

    try:
        engine = ctx["sweep_engine"]
        results = await run_cancellable_sweep(engine, sweep, progress_callback, is_cancelled)
        best_run_id = await service.save_sweep_results(sweep, sweep_id, results)
        validate_sweep_coverage(results)
    except SweepCancelled:
        await service.handle_sweep_cancelled(sweep_id)
        return
    except asyncio.CancelledError:
        await service.record_sweep_failure(sweep_id, "Task cancelled (timeout or worker shutdown)")
        raise
    except Exception as exc:
        cancelled = await service.record_sweep_failure(sweep_id, job_error_message(exc))

        if cancelled:
            return

        raise

    await service.complete_sweep(sweep_id, results, best_run_id)
    try:
        await service.save_sweep_reports(sweep_id, len(results))
    except Exception:
        logger.exception("Sweep reports deferred: sweep=%s", sweep_id)
