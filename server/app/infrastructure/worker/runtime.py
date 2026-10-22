"""Shared worker execution: billing, admission, heartbeat and job outcomes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from functools import wraps
import logging
from typing import TYPE_CHECKING, Any

from arq import Retry

from application.benchmark.sweep_jobs import IncompleteEvaluationError
from infrastructure.budget.job_context import job_billing_scope
from infrastructure.worker.admission import document_admission
from infrastructure.worker.outcomes import job_error_message


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from application.uow.factory import UnitOfWorkFactory

logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL_SEC = 60
MAX_JOB_TRIES = 3


@asynccontextmanager
async def job_heartbeat(
    uow_factory: UnitOfWorkFactory, job_id: int, interval_sec: float = HEARTBEAT_INTERVAL_SEC
) -> AsyncIterator[None]:
    """Periodically touch ``heartbeat_at`` while the job body runs."""

    async def beat() -> None:
        while True:
            await asyncio.sleep(interval_sec)
            try:
                async with uow_factory.create(master=True) as uow:
                    await uow.background_jobs.touch_heartbeat(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Heartbeat update failed for job %d", job_id)

    task = asyncio.create_task(beat(), name=f"job-heartbeat-{job_id}")
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def mark_job_failed_safe(uow_factory: UnitOfWorkFactory, job_id: int, error: str) -> None:
    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_failed(job_id, error[:500])
    except Exception:
        logger.exception("Worker: failed to mark job %d as failed", job_id)


async def run_tracked_job(
    uow_factory: UnitOfWorkFactory,
    job_id: int,
    action: Callable[[], Awaitable[None]],
    *,
    description: str,
    job_try: int = 1,
    max_tries: int = MAX_JOB_TRIES,
) -> None:
    """Run a job body with heartbeat, retries and durable job state."""
    async with job_heartbeat(uow_factory, job_id, interval_sec=HEARTBEAT_INTERVAL_SEC):
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_running(job_id)
        except Exception:
            logger.exception("Worker: failed to mark job %d as running", job_id)

        try:
            await action()
        except asyncio.CancelledError:
            logger.warning("Worker: job %d cancelled (%s) — persisting failed state", job_id, description)
            await mark_job_failed_safe(uow_factory, job_id, "Task cancelled (timeout or worker shutdown)")
            raise
        except IncompleteEvaluationError as exc:
            logger.warning("Worker: job %d incomplete (%s): %s", job_id, description, exc)
            await mark_job_failed_safe(uow_factory, job_id, str(exc))
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

            await mark_job_failed_safe(uow_factory, job_id, job_error_message(e))
        else:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_done(job_id)


def job(
    *, per_principal: bool = False, max_tries: int = MAX_JOB_TRIES
) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
    """Apply the worker lifecycle once, excluding admission waits from retries."""

    def decorate(function: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        @wraps(function)
        async def run(
            ctx: dict[str, Any],
            *,
            cost_principal: str | None = None,
            cost_operation_id: str | None = None,
            **kwargs: Any,  # noqa: ANN401 — forwards heterogeneous task arguments
        ) -> None:
            factory = ctx["container"].infrastructure.db.uow_factory
            with job_billing_scope(function.__name__, ctx, kwargs, cost_principal, cost_operation_id):
                async with document_admission(ctx, kwargs, enabled=per_principal) as run_ctx:

                    async def action() -> None:
                        await function(run_ctx, **kwargs)

                    await run_tracked_job(
                        factory,
                        kwargs["job_id"],
                        action,
                        description=function.__name__,
                        job_try=run_ctx.get("job_try", 1),
                        max_tries=max_tries,
                    )

        return run

    return decorate
