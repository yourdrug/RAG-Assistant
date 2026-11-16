"""Start and resume sweeps as tracked background jobs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from application.benchmark.dto import SweepCreateDTO
    from application.benchmark.services import BenchmarkSweepService
    from application.ports.job_enqueuer import JobEnqueuerPort
    from application.uow import UnitOfWorkFactory
    from domain.benchmark.sweep import BenchmarkSweep

logger = logging.getLogger(__name__)


class BenchmarkSweepLauncher:
    """Persist sweep and submission atomically; Redis delivery is best effort."""

    def __init__(
        self, uow_factory: UnitOfWorkFactory, sweeps: BenchmarkSweepService, enqueuer: JobEnqueuerPort
    ) -> None:
        self._uow_factory = uow_factory
        self._sweeps = sweeps
        self._enqueuer = enqueuer

    async def start(
        self, body: SweepCreateDTO, *, cost_principal: str | None = None
    ) -> tuple[BenchmarkSweep, int]:
        sweep = await self._sweeps.create(body, submit_job=True, cost_principal=cost_principal)
        return sweep, await self.enqueue(sweep)

    async def resume(self, sweep_id: int, *, cost_principal: str | None = None) -> tuple[BenchmarkSweep, int]:
        sweep = await self._sweeps.resume(sweep_id, submit_job=True, cost_principal=cost_principal)
        return sweep, await self.enqueue(sweep)

    async def enqueue(self, sweep: BenchmarkSweep) -> int:
        if sweep.id is None:
            message = "BenchmarkSweep saved with None id"
            raise RuntimeError(message)

        job_id = sweep.job_id

        if job_id is None:
            message = "Sweep has no durable job"
            raise RuntimeError(message)

        try:
            await self._enqueuer.enqueue_sweep(sweep_id=sweep.id, job_id=job_id)
        except Exception as exc:  # noqa: BLE001 — committed work is retried by the relay
            logger.warning(
                "Sweep delivery deferred: sweep_id=%d job_id=%d error_type=%s",
                sweep.id,
                job_id,
                type(exc).__name__,
            )
        return job_id
