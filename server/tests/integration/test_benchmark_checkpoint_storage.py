"""PostgreSQL recovery commits and idempotent per-config result persistence."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.entities.benchmark_run import BenchmarkRun
from domain.entities.benchmark_sweep import BenchmarkSweep
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.database.models import BenchmarkCheckpointModel, BenchmarkRunModel, BenchmarkSweepModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def checkpoint_factory(sessions):
    class Factory:
        @asynccontextmanager
        async def create(self, master=False):
            async with sessions.begin() as session:
                yield SimpleNamespace(benchmark_sweeps=SQLAlchemyBenchmarkSweepRepository(session))

    return Factory()


async def test_checkpoint_commit_survives_new_store_and_cascades_on_sweep_delete(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        sweep = await SQLAlchemyBenchmarkSweepRepository(session).create(BenchmarkSweep())
    store = DatabaseBenchmarkCheckpoints(checkpoint_factory(sessions), sweep.id)
    await store.save("case-1", {"answer": "saved", "scores": {"faithfulness": 0}})
    restored = DatabaseBenchmarkCheckpoints(checkpoint_factory(sessions), sweep.id)
    assert await restored.load("case-1") == {"answer": "saved", "scores": {"faithfulness": 0}}
    await restored.save("case-1", {"answer": "saved", "scores": {"faithfulness": 0, "relevancy": 8}})
    async with sessions.begin() as session:
        assert await session.scalar(select(func.count()).select_from(BenchmarkCheckpointModel)) == 1
        await session.execute(delete(BenchmarkSweepModel).where(BenchmarkSweepModel.id == sweep.id))
        assert await session.scalar(select(func.count()).select_from(BenchmarkCheckpointModel)) == 0


async def test_concurrent_sweep_result_save_does_not_duplicate_configuration(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        sweep = await SQLAlchemyBenchmarkSweepRepository(session).create(BenchmarkSweep())

    async def save(score):
        async with sessions.begin() as session:
            return await SQLAlchemyBenchmarkRunRepository(session).save_for_sweep(
                BenchmarkRun(
                    sweep_id=sweep.id,
                    config_json={"top_k": 5},
                    summary_metrics={"relevancy": score},
                    per_question_results=[{"answer": "a", "relevancy": score}],
                )
            )

    results = await asyncio.gather(save(8), save(9))
    assert results[0].id == results[1].id
    async with sessions.begin() as session:
        assert await session.scalar(select(func.count()).select_from(BenchmarkRunModel)) == 1


async def test_checkpoint_migration_roundtrip(pg_engine):
    import importlib

    migration = importlib.import_module(
        "infrastructure.database.migrations.versions.m3d4e5f6g7h8_benchmark_checkpoints"
    )

    def roundtrip(connection):
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
            migration.upgrade()

    async with pg_engine.begin() as conn:
        await conn.run_sync(roundtrip)
