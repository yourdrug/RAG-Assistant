"""PostgreSQL contracts for the single active sweep invariant."""

import asyncio
from contextlib import asynccontextmanager
from importlib import import_module
from types import SimpleNamespace

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.dto.benchmark_dto import SweepCreateDTO
from application.services.benchmark_services import BenchmarkSweepService
from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.exceptions import BusinessRuleViolation
from domain.value_objects.benchmark_strategy import BenchmarkStrategy
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from infrastructure.database.models import BenchmarkSweepModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_concurrent_service_create_has_one_winner(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    barrier = asyncio.Barrier(2)

    class ConcurrentRepository(SQLAlchemyBenchmarkSweepRepository):
        async def has_active(self):
            result = await super().has_active()
            await asyncio.wait_for(barrier.wait(), timeout=5)
            return result

    class Factory:
        @asynccontextmanager
        async def create(self, master=False):
            async with sessions.begin() as session:
                yield SimpleNamespace(benchmark_sweeps=ConcurrentRepository(session))

    service = BenchmarkSweepService(Factory())
    body = SweepCreateDTO(strategy=BenchmarkStrategy.GRID.value, search_space={})
    results = await asyncio.wait_for(
        asyncio.gather(service.create(body), service.create(body), return_exceptions=True),
        timeout=10,
    )
    assert sum(isinstance(result, BenchmarkSweep) for result in results) == 1
    assert sum(isinstance(result, BusinessRuleViolation) for result in results) == 1
    async with sessions.begin() as session:
        assert (await session.scalar(select(func.count()).select_from(BenchmarkSweepModel))) == 1
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        winner = next(result for result in results if isinstance(result, BenchmarkSweep))
        await repo.update_status(winner.id, BenchmarkSweepStatus.DONE.value)
        # Terminal history does not occupy the singleton slot.
        await repo.create(BenchmarkSweep())


async def test_unique_index_also_guards_status_updates(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        await repo.create(BenchmarkSweep())
        terminal = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus.DONE.value))
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await repo.update_status(terminal.id, BenchmarkSweepStatus.RUNNING.value)


async def test_concurrent_resume_has_one_winner(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        terminal = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus.FAILED.value))

    async def resume():
        async with sessions.begin() as session:
            return await SQLAlchemyBenchmarkSweepRepository(session).requeue(terminal.id)

    results = await asyncio.gather(resume(), resume())
    assert sorted(results) == [False, True]


async def test_cleanup_lock_serializes_with_resume(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        terminal = await SQLAlchemyBenchmarkSweepRepository(session).create(
            BenchmarkSweep(status=BenchmarkSweepStatus.FAILED.value)
        )

    async def resume():
        async with sessions.begin() as session:
            return await SQLAlchemyBenchmarkSweepRepository(session).requeue(terminal.id)

    async with sessions.begin() as session:
        locked = await SQLAlchemyBenchmarkSweepRepository(session).get_by_id(terminal.id, for_update=True)
        assert locked.status == BenchmarkSweepStatus.FAILED.value
        task = asyncio.create_task(resume())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
    assert await asyncio.wait_for(task, timeout=5) is True


async def test_resume_cannot_displace_an_active_sweep(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        await repo.create(BenchmarkSweep())
        terminal = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus.CANCELLED.value))
    async with sessions.begin() as session:
        with pytest.raises(BusinessRuleViolation):
            async with session.begin_nested():
                await SQLAlchemyBenchmarkSweepRepository(session).requeue(terminal.id)


async def test_dead_job_releases_active_sweep(pg_engine):
    from domain.value_objects.job_status import BackgroundJobStatus
    from presentation.api.constants import JobType
    from infrastructure.database.models import BackgroundJobModel

    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        job = BackgroundJobModel(job_type=JobType.SWEEP.value, status=BackgroundJobStatus.FAILED.value)
        session.add(job)
        await session.flush()
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        sweep = await repo.create(BenchmarkSweep(job_id=job.id, status=BenchmarkSweepStatus.RUNNING.value))
        assert await repo.fail_dead_jobs() == 1
        assert (await repo.get_by_id(sweep.id)).status == BenchmarkSweepStatus.FAILED.value
        assert await repo.requeue(sweep.id) is True
        # Requeue clears the old failed job; the reaper cannot fail the new attempt.
        assert await repo.fail_dead_jobs() == 0


async def test_migration_upgrade_and_downgrade(pg_engine):
    migration = import_module(
        "infrastructure.database.migrations.versions.h1c2d3e4f5g6_one_active_benchmark_sweep"
    )

    def migrate(conn, operation):
        with Operations.context(MigrationContext.configure(conn)):
            operation()

    async with pg_engine.begin() as conn:
        await conn.run_sync(lambda c: migrate(c, migration.downgrade))
        await conn.run_sync(lambda c: migrate(c, migration.upgrade))
        insert = text(
            "INSERT INTO benchmark_sweeps (strategy, search_space, objective_weights, creation_date) "
            "VALUES (:strategy, '{}', '{}', CURRENT_TIMESTAMP)"
        )
        params = {"strategy": BenchmarkStrategy.GRID.value}
        await conn.execute(insert, params)
        with pytest.raises(IntegrityError):
            async with conn.begin_nested():
                await conn.execute(insert, params)
        await conn.run_sync(lambda c: migrate(c, migration.downgrade))
        await conn.execute(insert, params)
        # Existing duplicates must fail migration without silently cancelling jobs.
        with pytest.raises(IntegrityError):
            async with conn.begin_nested():
                await conn.run_sync(lambda c: migrate(c, migration.upgrade))
