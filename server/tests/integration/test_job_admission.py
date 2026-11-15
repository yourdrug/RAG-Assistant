"""Admission locks isolate principals across concurrent API transactions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from config import JobsSection
from domain.runtime.jobs import BackgroundJob, JobAdmissionError, JobType
from infrastructure.database.models import BackgroundJobModel
from infrastructure.repositories.misc import sqlalchemy_background_job_repository as jobs_module


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def pending(principal: str | None) -> BackgroundJob:
    return BackgroundJob(job_type=JobType.INGEST_FILE, dispatch_payload={"cost_principal": principal})


@pytest.mark.parametrize("second_principal", ["user:2", None])
async def test_other_principal_does_not_wait_for_uncommitted_upload(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, second_principal: str | None
) -> None:
    limits = JobsSection(job_backlog_limit=1, job_principal_backlog_limit=1)
    monkeypatch.setattr(jobs_module, "process_section", lambda _: limits)
    sessions = async_sessionmaker(pg_engine)
    async with sessions() as first, sessions() as second:
        await jobs_module.SQLAlchemyBackgroundJobRepository(first).create(pending("user:1"))
        # The global overload guard may overshoot while another principal is uncommitted.
        await asyncio.wait_for(
            jobs_module.SQLAlchemyBackgroundJobRepository(second).create(pending(second_principal)),
            timeout=2,
        )
        await second.commit()
        await first.commit()

    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(BackgroundJobModel)) == 2
        with pytest.raises(JobAdmissionError) as error:
            await jobs_module.SQLAlchemyBackgroundJobRepository(session).create(pending("user:3"))
        assert error.value.scope == "global"


@pytest.mark.parametrize("commit_first", [True, False])
async def test_same_principal_serializes_until_commit_or_rollback(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, commit_first: bool
) -> None:
    limits = JobsSection(job_backlog_limit=10, job_principal_backlog_limit=1)
    monkeypatch.setattr(jobs_module, "process_section", lambda _: limits)
    sessions = async_sessionmaker(pg_engine)
    async with sessions() as first, sessions() as second:
        await jobs_module.SQLAlchemyBackgroundJobRepository(first).create(pending("user:1"))
        task = asyncio.create_task(
            jobs_module.SQLAlchemyBackgroundJobRepository(second).create(pending("user:1"))
        )
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.1)

            if commit_first:
                await first.commit()
                with pytest.raises(JobAdmissionError) as error:
                    await asyncio.wait_for(task, timeout=2)
                assert error.value.scope == "principal"
            else:
                await first.rollback()
                job = await asyncio.wait_for(task, timeout=2)
                assert job.id is not None
                await second.commit()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
