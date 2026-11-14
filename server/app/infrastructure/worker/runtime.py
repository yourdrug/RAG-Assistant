"""Shared worker execution: principal, heartbeat and job outcomes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from functools import wraps
import logging
import random
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from arq import Retry

from application.benchmark.sweep_jobs import IncompleteEvaluationError
from application.jobs.execution import ExecutionLeaseLostError, JobExecution, current_execution
from application.jobs.principal import job_principal
from domain.runtime.jobs import JobConcurrencyLimitReachedError
from infrastructure.metrics.jobs import FencingMeasurements, fencing_measurements, record_job_event
from infrastructure.worker.outcomes import job_error_message
from infrastructure.worker.policy import (
    DEFAULT_MAX_TRIES,
    RetryPolicy,
    concurrency_limit,
    max_tries as policy_max_tries,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine

    from application.uow.factory import UnitOfWorkFactory
    from domain.runtime.jobs import ConcurrencyLimit

logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL_SEC = 60
MAX_JOB_TRIES = DEFAULT_MAX_TRIES
LEASE_SECONDS = 180
WORKLOAD_BUSY_DEFER_SEC = 15


async def keep_lease(
    uow_factory: UnitOfWorkFactory, job_id: int, attempt_id: str, interval_sec: float
) -> None:
    """Renew the lease; return only once it is lost or can no longer be confirmed.

    A transient renewal error is tolerated while the last successful renewal
    still covers the lease, so a database blip does not kill a long job.
    """
    renewed_at = time.monotonic()

    while True:
        await asyncio.sleep(interval_sec)
        try:
            async with uow_factory.create(master=True) as uow:
                renewed = await uow.background_jobs.renew_lease(
                    job_id, attempt_id, lease_seconds=LEASE_SECONDS
                )
        except Exception:
            if time.monotonic() - renewed_at + interval_sec < LEASE_SECONDS:
                logger.warning("Heartbeat update failed for job %d; retrying", job_id, exc_info=True)
                continue

            logger.exception("Heartbeat update failed for job %d; lease lapsing, stopping", job_id)
            return

        if not renewed:
            logger.error("Execution lease lost for job %d", job_id)
            return

        renewed_at = time.monotonic()


@asynccontextmanager
async def job_heartbeat(
    uow_factory: UnitOfWorkFactory, job_id: int, attempt_id: str, interval_sec: float = HEARTBEAT_INTERVAL_SEC
) -> AsyncIterator[None]:
    """Keep the execution lease while the job body runs; cancel the body once it is lost."""
    owner = asyncio.current_task()

    async def beat() -> None:
        await keep_lease(uow_factory, job_id, attempt_id, interval_sec)

        if owner is not None:
            owner.cancel()

    task = asyncio.create_task(beat(), name=f"job-heartbeat-{job_id}")
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def finish_attempt(
    uow_factory: UnitOfWorkFactory,
    job_id: int,
    attempt_id: str,
    error: str | None = None,
    retry_seconds: float | None = None,
) -> bool:
    """Persist the outcome of the current attempt; False when it was superseded.

    Database errors propagate: the lease then expires and recovery decides.
    """
    async with uow_factory.create(master=True) as uow:
        return await uow.background_jobs.finish_attempt(
            job_id, attempt_id, error=error[:500] if error else None, retry_seconds=retry_seconds
        )


async def claim_attempt(
    uow_factory: UnitOfWorkFactory,
    job_id: int,
    attempt_id: str,
    limit: ConcurrencyLimit | None,
    job_type: str = "unknown",
) -> int | None:
    """Return the new attempt number, or None when the job is not due/claimable."""
    try:
        async with uow_factory.create(master=True) as uow:
            return await uow.background_jobs.claim(
                job_id, attempt_id, lease_seconds=LEASE_SECONDS, limit=limit
            )
    except JobConcurrencyLimitReachedError as exc:
        await record_job_event("deferred", job_type)
        # Waiting in-process would hold a broker slot and burn the task timeout
        # before the job starts; hand the delivery back without using an attempt.
        logger.info("Job %d deferred: concurrency group full (%s)", job_id, exc)
        defer = WORKLOAD_BUSY_DEFER_SEC + random.uniform(0, WORKLOAD_BUSY_DEFER_SEC)  # noqa: S311
        raise Retry(defer=defer) from exc


async def run_tracked_job(  # noqa: C901, PLR0913, PLR0915 -- claim and outcome form one lifecycle
    uow_factory: UnitOfWorkFactory,
    job_id: int,
    action: Callable[[], Awaitable[None]],
    *,
    description: str,
    job_try: int = 1,
    max_tries: int = MAX_JOB_TRIES,
    limit: ConcurrencyLimit | None = None,
    job_type: str = "unknown",
) -> None:
    """Claim before any effect; only the current attempt may finish or retry."""
    del job_try  # Broker delivery attempts are independent of execution attempts.
    attempt_id = str(uuid4())
    attempt = await claim_attempt(uow_factory, job_id, attempt_id, limit, job_type)

    if attempt is None:
        logger.info("Skipping duplicate, terminal or deferred job %d", job_id)
        return

    if attempt > max_tries:
        await finish_attempt(uow_factory, job_id, attempt_id, "Execution attempts exhausted")
        return

    async with job_heartbeat(uow_factory, job_id, attempt_id, interval_sec=HEARTBEAT_INTERVAL_SEC):
        try:
            token = current_execution.set(JobExecution(job_id, attempt_id))
            measurements = FencingMeasurements()
            measurements_token = fencing_measurements.set(measurements)
            try:
                await action()
            finally:
                current_execution.reset(token)
                fencing_measurements.reset(measurements_token)
                logger.info(
                    "Job fencing: job_id=%d checks=%d seconds=%.6f",
                    job_id,
                    measurements.checks,
                    measurements.seconds,
                )
        except asyncio.CancelledError:
            # Worker shutdown (e.g. a deploy) must not burn the job: release the
            # attempt for redelivery while the execution budget allows it.
            delay = RetryPolicy(max_tries=max_tries).backoff(attempt) if attempt < max_tries else None
            try:
                await finish_attempt(
                    uow_factory, job_id, attempt_id, "Task cancelled (timeout or worker shutdown)", delay
                )
            except Exception:
                # Keep the cancellation; the expired lease is recovered by cron.
                logger.exception("Could not persist cancellation of job %d", job_id)
            raise
        except ExecutionLeaseLostError:
            logger.warning("Discarding superseded job attempt %d", job_id)
        except IncompleteEvaluationError as exc:
            logger.warning("Worker: job %d incomplete (%s)", job_id, description)
            await finish_attempt(uow_factory, job_id, attempt_id, str(exc))
        except Exception as exc:
            policy = RetryPolicy(max_tries=max_tries)
            delay = policy.delay(exc, attempt) if attempt < max_tries else None
            saved = await finish_attempt(uow_factory, job_id, attempt_id, job_error_message(exc), delay)
            logger.warning(
                "Job failed: id=%d task=%s attempt=%d error_type=%s retry=%s",
                job_id,
                description,
                attempt,
                type(exc).__name__,
                delay is not None,
            )

            if saved and delay is not None:
                raise Retry(defer=delay) from exc
        else:
            await finish_attempt(uow_factory, job_id, attempt_id)


def job(
    job_type: str, *, max_tries: int | None = None
) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Coroutine[Any, Any, None]]]:
    """Apply principal, heartbeat, retry and concurrency policy of ``job_type`` to a worker task."""
    tries = max_tries if max_tries is not None else policy_max_tries(job_type)

    def decorate(function: Callable[..., Awaitable[None]]) -> Callable[..., Coroutine[Any, Any, None]]:
        @wraps(function)
        async def run(
            ctx: dict[str, Any],
            *,
            cost_principal: str | None = None,
            **kwargs: Any,  # noqa: ANN401 — forwards heterogeneous task arguments
        ) -> None:
            token = job_principal.set(cost_principal)
            try:

                async def action() -> None:
                    await function(ctx, **kwargs)

                await run_tracked_job(
                    ctx["uow_factory"],
                    kwargs["job_id"],
                    action,
                    description=function.__name__,
                    job_try=ctx.get("job_try", 1),
                    max_tries=tries,
                    limit=concurrency_limit(job_type),
                    job_type=job_type,
                )
            finally:
                job_principal.reset(token)

        return run

    return decorate
