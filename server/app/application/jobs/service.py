"""Application service for background job management."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus, JobType
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError


if TYPE_CHECKING:
    from application.jobs.ports import JobAbortPort
    from application.uow.factory import UnitOfWorkFactory


logger = logging.getLogger("default")


class JobService:
    def __init__(self, uow_factory: UnitOfWorkFactory, *, aborter: JobAbortPort | None = None) -> None:
        self._uow_factory = uow_factory
        self.aborter = aborter

    async def check_capacity(self, principal: str) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.background_jobs.check_admission(principal)

    async def cancel(self, job_id: int, *, user_id: int, is_admin: bool = False) -> None:
        async with self._uow_factory.create(master=True) as uow:
            job = await uow.background_jobs.get_by_id(job_id)

            # A foreign job is reported as missing so its id cannot be probed.
            if job is None or (
                not is_admin and (job.dispatch_payload or {}).get("cost_principal") != f"user:{user_id}"
            ):
                entity_name = "BackgroundJob"
                raise EntityNotFoundError(entity_name, job_id)

            if job.job_type not in {JobType.INGEST, JobType.INGEST_FILE, JobType.BENCHMARK}:
                message = "This task has no cancellation action"
                raise BusinessRuleViolationError(message)

            changed = await uow.background_jobs.cancel(job_id)

        # Commit cancellation before touching Redis: relay and old owners are fenced.
        if changed and self.aborter is not None:
            try:
                await self.aborter.abort_job(job_id, job.job_type, job.related_id)
            except (ConnectionError, OSError, TimeoutError):
                logger.warning("Broker abort deferred for cancelled job %d", job_id)

    async def retry(self, job_id: int) -> int:
        async with self._uow_factory.create(master=True) as uow:
            old = await uow.background_jobs.get_by_id(job_id)

            if old is None:
                entity_name = "BackgroundJob"
                raise EntityNotFoundError(entity_name, job_id)

            if (
                old.status not in {BackgroundJobStatus.FAILED, BackgroundJobStatus.CANCELLED}
                or old.dispatch_payload is None
            ):
                message = "Only failed or cancelled durable tasks can be retried"
                raise BusinessRuleViolationError(message)

            if old.job_type == JobType.SWEEP:
                message = "Use the sweep resume action"
                raise BusinessRuleViolationError(message)

            new = await uow.background_jobs.create(
                BackgroundJob(
                    job_type=old.job_type,
                    related_id=old.related_id,
                    dispatch_payload=dict(old.dispatch_payload),
                    payload_version=old.payload_version,
                    deduplication_key=old.deduplication_key or f"retry:{job_id}",
                    progress=None if old.progress is None else dict(old.progress),
                )
            )

            if new.id is None:
                message = "BackgroundJob create returned None id"
                raise RuntimeError(message)

            return new.id

    async def create_job(
        self,
        job_type: str,
        related_id: int | None = None,
        *,
        dispatch_payload: dict[str, object] | None = None,
        cost_principal: str | None = None,
    ) -> int:
        """Create a background job record and return its ID."""
        payload = None if dispatch_payload is None else {**dispatch_payload, "cost_principal": cost_principal}
        deduplication_key = None

        if dispatch_payload is not None and job_type in {JobType.INGEST, JobType.INGEST_FILE}:
            deduplication_key = hashlib.sha256(
                json.dumps([job_type, dispatch_payload], sort_keys=True).encode()
            ).hexdigest()

        job = BackgroundJob(
            job_type=job_type,
            status=BackgroundJobStatus.PENDING,
            related_id=related_id,
            dispatch_payload=payload,
            deduplication_key=deduplication_key,
        )
        async with self._uow_factory.create(master=True) as uow:
            created = await uow.background_jobs.create(job)

            if created.id is None:
                msg = "BackgroundJob create returned None id"
                raise RuntimeError(msg)

            return created.id

    async def mark_failed(self, job_id: int, error: str) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_failed(job_id, error)

    async def get(self, job_id: int) -> BackgroundJob:
        async with self._uow_factory.create() as uow:
            job = await uow.background_jobs.get_by_id(job_id)

        if job is None:
            entity_name = "BackgroundJob"
            raise EntityNotFoundError(entity_name, job_id)

        return job

    async def list_recent(
        self, limit: int = 50, offset: int = 0, status: str | None = None
    ) -> list[BackgroundJob]:
        async with self._uow_factory.create() as uow:
            if status is not None:
                return await uow.background_jobs.list_recent(limit=limit, offset=offset, status=status)

            return await uow.background_jobs.list_recent(limit=limit, offset=offset)

    async def count_by_status(self) -> dict[str, int]:
        async with self._uow_factory.create() as uow:
            return await uow.background_jobs.count_by_status()
