"""PostgreSQL recovery commits and idempotent per-config result persistence."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.integration.uow import session_factory

from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweep
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.database.models import BenchmarkCheckpointModel, BenchmarkRunModel, BenchmarkSweepModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from infrastructure.uow_factory import UnitOfWorkFactory


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def checkpoint_factory(sessions: async_sessionmaker[AsyncSession]) -> UnitOfWorkFactory:

    return session_factory(sessions)


async def test_checkpoint_commit_survives_new_store_and_cascades_on_sweep_delete(
    pg_engine: AsyncEngine,
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        sweep = await SQLAlchemyBenchmarkSweepRepository(session).create(BenchmarkSweep())
        assert sweep.id is not None
    store = DatabaseBenchmarkCheckpoints(checkpoint_factory(sessions), sweep.id)
    await store.save("case-1", {"answer": "saved", "scores": {"faithfulness": 0}})
    restored = DatabaseBenchmarkCheckpoints(checkpoint_factory(sessions), sweep.id)
    assert await restored.load("case-1") == {"answer": "saved", "scores": {"faithfulness": 0}}
    await restored.save("case-1", {"answer": "saved", "scores": {"faithfulness": 0, "relevancy": 8}})
    async with sessions.begin() as session:
        assert await session.scalar(select(func.count()).select_from(BenchmarkCheckpointModel)) == 1
        await session.execute(delete(BenchmarkSweepModel).where(BenchmarkSweepModel.id == sweep.id))
        assert await session.scalar(select(func.count()).select_from(BenchmarkCheckpointModel)) == 0


async def test_concurrent_sweep_result_save_does_not_duplicate_configuration(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        sweep = await SQLAlchemyBenchmarkSweepRepository(session).create(BenchmarkSweep())
        assert sweep.id is not None

    async def save(score: int) -> BenchmarkRun:
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
