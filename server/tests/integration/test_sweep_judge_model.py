"""PostgreSQL persistence and reversible migration for the selected judge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.fakes import require_present

from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_selected_judge_round_trips_through_repository(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        created = await repo.create(BenchmarkSweep(judge_model="chosen/judge"))
        sweep_id = created.id
        assert created.judge_model == "chosen/judge"
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        assert require_present(await repo.get_by_id(require_present(sweep_id))).judge_model == "chosen/judge"
        assert (await repo.list_items())[0].judge_model == "chosen/judge"


async def test_judge_default_and_updates(pg_engine: AsyncEngine) -> None:

    async with pg_engine.begin() as conn:
        sweep_id = (
            await conn.execute(
                text(
                    "INSERT INTO benchmark_sweeps "
                    "(strategy, search_space, objective_weights, status, creation_date) "
                    "VALUES ('grid', '{}', '{}', :status, CURRENT_TIMESTAMP) RETURNING id"
                ),
                {"status": BenchmarkSweepStatus.DONE.value},
            )
        ).scalar_one()
        legacy = (
            await conn.execute(
                text("SELECT status, judge_model FROM benchmark_sweeps WHERE id=:id"), {"id": sweep_id}
            )
        ).one()
        assert legacy == (BenchmarkSweepStatus.DONE.value, None)
        await conn.execute(
            text("UPDATE benchmark_sweeps SET judge_model=:model WHERE id=:id"),
            {"model": "chosen/judge", "id": sweep_id},
        )
        assert (
            await conn.execute(text("SELECT status FROM benchmark_sweeps WHERE id=:id"), {"id": sweep_id})
        ).scalar_one() == BenchmarkSweepStatus.DONE.value
