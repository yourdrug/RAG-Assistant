"""Start and resume sweeps as tracked background jobs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.benchmark.sweep import BenchmarkSweepStatus
from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus, JobType


if TYPE_CHECKING:
    from application.benchmark.dto import SweepCreateDTO
    from application.benchmark.services import BenchmarkSweepService
    from application.ports.job_enqueuer import JobEnqueuerPort
    from application.uow import UnitOfWorkFactory
    from domain.benchmark.sweep import BenchmarkSweep

logger = logging.getLogger(__name__)


class BenchmarkSweepLauncher:
    """A sweep that cannot be queued is marked failed, so it neither blocks new sweeps nor stays pending."""

    def __init__(
        self, uow_factory: UnitOfWorkFactory, sweeps: BenchmarkSweepService, enqueuer: JobEnqueuerPort
    ) -> None:
        self._uow_factory = uow_factory
        self._sweeps = sweeps
        self._enqueuer = enqueuer

    async def start(self, body: SweepCreateDTO) -> tuple[BenchmarkSweep, int]:
        sweep = await self._sweeps.create(body)
        return sweep, await self.enqueue(sweep)

    async def resume(self, sweep_id: int) -> tuple[BenchmarkSweep, int]:
        sweep = await self._sweeps.resume(sweep_id)
        return sweep, await self.enqueue(sweep)

    async def enqueue(self, sweep: BenchmarkSweep) -> int:
        if sweep.id is None:
            message = "BenchmarkSweep saved with None id"
            raise RuntimeError(message)

        sweep_id = sweep.id
        job_id: int | None = None

        try:
            job_id = await self.create_job(sweep_id)
            await self._enqueuer.enqueue_sweep(sweep_id=sweep_id, job_id=job_id)
        except Exception:
            await self.mark_unqueued(sweep_id, job_id)
            raise

        return job_id

    async def create_job(self, sweep_id: int) -> int:
        """Commit the job together with its link to the sweep, before the queue is contacted."""
        async with self._uow_factory.create(master=True) as uow:
            job = await uow.background_jobs.create(
                BackgroundJob(job_type=JobType.SWEEP, status=BackgroundJobStatus.PENDING, related_id=sweep_id)
            )

            if job.id is None:
                message = "BackgroundJob create returned None id"
                raise RuntimeError(message)

            await uow.benchmark_sweeps.set_job_id(sweep_id, job.id)
            return job.id

    async def mark_unqueued(self, sweep_id: int, job_id: int | None) -> None:
        try:
            async with self._uow_factory.create(master=True) as uow:
                if job_id is not None:
                    await uow.background_jobs.mark_failed(job_id, "Failed to enqueue benchmark sweep")

                await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.FAILED.value)
        except Exception:
            # The original enqueue error is re-raised by the caller.
            logger.exception("Could not mark sweep %d failed after an enqueue error", sweep_id)
