"""Shared utilities for route modules."""

from __future__ import annotations

import logging

from shared import request_id_ctx
from domain.repositories.background_job_repository import BackgroundJob
from domain.value_objects.job_status import BackgroundJobStatus
from infrastructure.uow_factory import UnitOfWorkFactory  # noqa: F401

logger = logging.getLogger("default")


async def create_background_job(
    uow_factory: UnitOfWorkFactory, job_type: str, related_id: int | None = None
) -> int:
    """Create a background job record and return its ID."""
    async with uow_factory.create(master=True) as uow:
        job = BackgroundJob(
            job_type=job_type,
            status=BackgroundJobStatus.PENDING.value,
            related_id=related_id,
            request_id=request_id_ctx.get("-"),
        )
        job = await uow.background_jobs.create(job)
        return job.id  # type: ignore[return-value]


def safe_background_call(func, *args, **kwargs):
    """Run a sync function in a background task, catching and logging exceptions."""
    try:
        func(*args, **kwargs)
    except Exception:
        logger.exception("Background task failed")
