"""Evaluation mode persistence, DB invariants and reversible migration."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.benchmark.sweep import BenchmarkSweep
from domain.benchmark.sweep import BenchmarkStrategy
from domain.benchmark.sweep import SweepEvaluationMode
from domain.benchmark.sweep import BenchmarkSweepStatus
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize("mode", list(SweepEvaluationMode))
async def test_mode_round_trips_through_repository(pg_engine, mode):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        created = await repo.create(BenchmarkSweep(evaluation_mode=mode.value, top_n_llm=0))
        assert created.evaluation_mode == mode
        sweep_id = created.id
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        assert (await repo.get_by_id(sweep_id)).evaluation_mode == mode
        assert (await repo.list_items())[0].evaluation_mode == mode


@pytest.mark.parametrize(
    "mode,strategy",
    [("invalid", BenchmarkStrategy.GRID), (SweepEvaluationMode.FULL, BenchmarkStrategy.RANDOM)],
)
async def test_database_rejects_invalid_modes_and_nonexhaustive_full_search(pg_engine, mode, strategy):
    async with pg_engine.connect() as conn:
        with pytest.raises(IntegrityError):
            await conn.execute(
                text(
                    "INSERT INTO benchmark_sweeps "
                    "(strategy, search_space, objective_weights, evaluation_mode, creation_date) "
                    "VALUES (:strategy, '{}', '{}', :mode, CURRENT_TIMESTAMP)"
                ),
                {"strategy": strategy.value, "mode": str(mode)},
            )


async def test_evaluation_mode_default_and_updates(pg_engine):

    async with pg_engine.begin() as conn:
        sweep_id = (
            await conn.execute(
                text(
                    "INSERT INTO benchmark_sweeps "
                    "(strategy, search_space, objective_weights, status, creation_date) "
                    "VALUES (:strategy, '{}', '{}', :status, CURRENT_TIMESTAMP) RETURNING id"
                ),
                {"strategy": BenchmarkStrategy.GRID.value, "status": BenchmarkSweepStatus.DONE.value},
            )
        ).scalar_one()
        assert (
            await conn.execute(
                text("SELECT status, evaluation_mode FROM benchmark_sweeps WHERE id=:id"), {"id": sweep_id}
            )
        ).one() == (BenchmarkSweepStatus.DONE.value, SweepEvaluationMode.FAST.value)
        await conn.execute(
            text("UPDATE benchmark_sweeps SET evaluation_mode=:mode WHERE id=:id"),
            {"id": sweep_id, "mode": SweepEvaluationMode.FULL.value},
        )
        assert (
            await conn.execute(text("SELECT status FROM benchmark_sweeps WHERE id=:id"), {"id": sweep_id})
        ).scalar_one() == BenchmarkSweepStatus.DONE.value
