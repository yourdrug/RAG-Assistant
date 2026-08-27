"""Application service for background job management."""

from __future__ import annotations

import logging

from domain.entities.background_job import BackgroundJob
from domain.value_objects.job_status import BackgroundJobStatus

from application.ports.unit_of_work_factory import UnitOfWorkFactory

logger = logging.getLogger("default")


class JobService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def create_job(self, job_type: str, related_id: int | None = None) -> int:
        """Create a background job record and return its ID."""
        job = BackgroundJob(
            job_type=job_type,
            status=BackgroundJobStatus.PENDING.value,
            related_id=related_id,
        )
        async with self._uow_factory.create(master=True) as uow:
            created = await uow.background_jobs.create(job)
            assert created.id is not None
            return created.id

    async def list_recent(self, limit: int = 50, offset: int = 0):
        async with self._uow_factory.create() as uow:
            return await uow.background_jobs.list_recent(limit=limit, offset=offset)

    async def count_by_status(self) -> dict[str, int]:
        async with self._uow_factory.create() as uow:
            return await uow.background_jobs.count_by_status()
