"""PostgreSQL contracts for the single active sweep invariant."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.integration.uow import SessionDatabase

from application.benchmark import BenchmarkSweepService, SweepCreateDTO
from domain.benchmark.sweep import BenchmarkStrategy, BenchmarkSweep, BenchmarkSweepStatus
from domain.runtime.jobs import BackgroundJobStatus
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.database.models import BackgroundJobModel, BenchmarkSweepModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)
from infrastructure.uow_factory import UnitOfWorkFactory
from presentation.api.constants import JobType


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

    from application.uow.unit_of_work import UnitOfWork


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_concurrent_service_create_has_one_winner(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    barrier = asyncio.Barrier(2)

    class ConcurrentRepository(SQLAlchemyBenchmarkSweepRepository):
        async def has_active(self) -> bool:
            result = await super().has_active()
            await asyncio.wait_for(barrier.wait(), timeout=5)
            return result

    class Factory(UnitOfWorkFactory):
        @asynccontextmanager
        async def create(self, *, master: bool = False) -> AsyncIterator[UnitOfWork]:
            async with super().create(master=master) as uow:
                assert isinstance(uow._session, AsyncSession)
                uow.benchmark_sweeps = ConcurrentRepository(uow._session)
                yield uow

    service = BenchmarkSweepService(Factory(SessionDatabase(sessions)))
    body = SweepCreateDTO(strategy=BenchmarkStrategy.GRID.value, search_space={})
    results = await asyncio.wait_for(
        asyncio.gather(service.create(body), service.create(body), return_exceptions=True),
        timeout=10,
    )
    assert sum(isinstance(result, BenchmarkSweep) for result in results) == 1
    assert sum(isinstance(result, BusinessRuleViolationError) for result in results) == 1
    async with sessions.begin() as session:
        assert (await session.scalar(select(func.count()).select_from(BenchmarkSweepModel))) == 1
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        winner = next(result for result in results if isinstance(result, BenchmarkSweep))
        assert winner.id is not None
        await repo.update_status(winner.id, BenchmarkSweepStatus.RUNNING)
        await repo.update_status(winner.id, BenchmarkSweepStatus.DONE)
        # Terminal history does not occupy the singleton slot.
        await repo.create(BenchmarkSweep())


async def test_unique_index_also_guards_status_updates(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        await repo.create(BenchmarkSweep())
        terminal = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus.DONE))
        assert terminal.id is not None
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await repo.update_status(terminal.id, BenchmarkSweepStatus.PENDING)


async def test_concurrent_resume_has_one_winner(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        terminal = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus.FAILED))
        assert terminal.id is not None

    async def resume() -> bool:
        async with sessions.begin() as session:
            assert terminal.id is not None
            return await SQLAlchemyBenchmarkSweepRepository(session).requeue(terminal.id)

    results = await asyncio.gather(resume(), resume())
    assert sorted(results) == [False, True]


async def test_cleanup_lock_serializes_with_resume(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        terminal = await SQLAlchemyBenchmarkSweepRepository(session).create(
            BenchmarkSweep(status=BenchmarkSweepStatus.FAILED)
        )
        assert terminal.id is not None

    async def resume() -> bool:
        async with sessions.begin() as session:
            assert terminal.id is not None
            return await SQLAlchemyBenchmarkSweepRepository(session).requeue(terminal.id)

    async with sessions.begin() as session:
        locked = await SQLAlchemyBenchmarkSweepRepository(session).get_by_id(terminal.id, for_update=True)
        assert locked is not None
        assert locked.status == BenchmarkSweepStatus.FAILED
        task = asyncio.create_task(resume())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
    assert await asyncio.wait_for(task, timeout=5) is True


async def test_resume_cannot_displace_an_active_sweep(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        await repo.create(BenchmarkSweep())
        terminal = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus.CANCELLED))
        assert terminal.id is not None
    async with sessions.begin() as session:
        with pytest.raises(BusinessRuleViolationError):
            async with session.begin_nested():
                await SQLAlchemyBenchmarkSweepRepository(session).requeue(terminal.id)


async def test_dead_job_releases_active_sweep(pg_engine: AsyncEngine) -> None:

    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        job = BackgroundJobModel(job_type=JobType.SWEEP.value, status=BackgroundJobStatus.FAILED.value)
        session.add(job)
        await session.flush()
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        sweep = await repo.create(BenchmarkSweep(job_id=job.id, status=BenchmarkSweepStatus.RUNNING))
        assert sweep.id is not None
        assert await repo.fail_dead_jobs() == 1
        failed = await repo.get_by_id(sweep.id)
        assert failed is not None
        assert failed.status == BenchmarkSweepStatus.FAILED
        assert await repo.requeue(sweep.id) is True
        # Requeue clears the old failed job; the reaper cannot fail the new attempt.
        assert await repo.fail_dead_jobs() == 0
