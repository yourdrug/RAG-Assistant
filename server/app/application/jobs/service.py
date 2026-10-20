"""Application service for background job management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.entities.background_job import BackgroundJob
from domain.shared.exceptions import EntityNotFound
from domain.value_objects.job_status import BackgroundJobStatus


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory


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

            if created.id is None:
                msg = "BackgroundJob create returned None id"
                raise RuntimeError(msg)

            return created.id

    async def get(self, job_id: int) -> BackgroundJob:
        async with self._uow_factory.create() as uow:
            job = await uow.background_jobs.get_by_id(job_id)

        if job is None:
            entity_name = "BackgroundJob"
            raise EntityNotFound(entity_name, job_id)

        return job

    async def list_recent(self, limit: int = 50, offset: int = 0) -> list[BackgroundJob]:
        async with self._uow_factory.create() as uow:
            return await uow.background_jobs.list_recent(limit=limit, offset=offset)

    async def count_by_status(self) -> dict[str, int]:
        async with self._uow_factory.create() as uow:
            return await uow.background_jobs.count_by_status()
