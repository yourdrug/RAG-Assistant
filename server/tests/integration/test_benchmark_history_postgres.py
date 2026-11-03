"""PostgreSQL run identity, compatible baselines, export and checkpoint retention."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.integration.uow import session_factory

from application.benchmark import BenchmarkExportService
from composition.service_providers import build_sweep_engine
from config import settings
from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.database.models import BenchmarkCheckpointModel, BenchmarkRunModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

    from application.benchmark.sweep import SweepEngine


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_job_and_import_retry_are_idempotent_and_do_not_mix_sweeps(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    async def save() -> BenchmarkRun:
        async with sessions.begin() as session:
            return await SQLAlchemyBenchmarkRunRepository(session).save_once(BenchmarkRun(job_id=55))

    saved = await asyncio.gather(save(), save())
    assert saved[0].id == saved[1].id
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        imported = await repo.save_once(BenchmarkRun(legacy_hash="a" * 64))
        repeated = await repo.save_once(BenchmarkRun(legacy_hash="a" * 64))
        assert imported.id == repeated.id
        assert await session.scalar(select(func.count()).select_from(BenchmarkRunModel)) == 2


async def test_baseline_rejects_other_dataset_mode_and_corpus(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        base = await repo.create(
            BenchmarkRun(
                dataset="legal",
                llm_evaluated=True,
                config_json={"evaluation_mode": "full", "collection_name": "a"},
            )
        )
        assert base.id is not None
        await repo.create(
            BenchmarkRun(
                dataset="other",
                llm_evaluated=True,
                config_json={"evaluation_mode": "full", "collection_name": "a"},
            )
        )
        await repo.create(
            BenchmarkRun(
                dataset="legal",
                llm_evaluated=True,
                config_json={"evaluation_mode": "fast", "collection_name": "a"},
            )
        )
        await repo.create(
            BenchmarkRun(
                dataset="legal",
                llm_evaluated=True,
                config_json={"evaluation_mode": "full", "collection_name": "b"},
            )
        )
        current = await repo.create(
            BenchmarkRun(
                dataset="legal",
                llm_evaluated=True,
                config_json={"evaluation_mode": "full", "collection_name": "a"},
            )
        )
        assert current.id is not None
        baseline = await repo.get_baseline(current)
        assert baseline is not None
        assert baseline.id == base.id


async def test_export_pages_real_json_array_and_nullable_legacy_results(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        run = await repo.create(BenchmarkRun(per_question_results=[{"id": n} for n in range(4)]))
        assert run.id is not None
        legacy = await repo.create(BenchmarkRun())
        assert legacy.id is not None
        assert await repo.result_page(run.id, offset=1, limit=2) == [{"id": 1}, {"id": 2}]
        assert await repo.result_page(legacy.id, offset=0, limit=1) == []

    data = b"".join(
        [block async for block in BenchmarkExportService(session_factory(sessions)).export(run.id, "json")]
    )
    assert json.loads(data)["results"] == [{"id": n} for n in range(4)]
    uploaded = []

    class Storage:
        async def upload_stream(self, key: str, blocks: AsyncIterator[bytes]) -> None:
            uploaded.append((key, b"".join([block async for block in blocks])))

    assert (
        await BenchmarkExportService(session_factory(sessions), MagicMock(wraps=Storage())).save_report(
            run.id, "json"
        )
        is True
    )
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        saved = await repo.get_by_id(run.id)
        assert saved is not None
        assert saved.artifacts["json"]["status"] == "ready"
        assert saved.artifacts["json"]["size_bytes"] == len(uploaded[0][1])
        assert saved.per_question_results == [{"id": n} for n in range(4)]


async def test_db_retention_preserves_active_and_recent_failed_checkpoints(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        sweeps = []

        for status in ("running", "failed", "cancelled", "done"):
            sweep = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus(status)))
            assert sweep.id is not None
            sweeps.append(sweep)
            await repo.save_checkpoint(sweep.id, "dataset.json", [])

        # Failed is recent; cancelled is old. Done is immediately disposable.
        await session.execute(
            update(BenchmarkCheckpointModel)
            .where(BenchmarkCheckpointModel.sweep_id.in_([sweeps[0].id, sweeps[2].id]))
            .values(updated_at=datetime.now(UTC) - timedelta(days=2))
        )
        assert await repo.cleanup_checkpoints(datetime.now(UTC) - timedelta(days=1)) == 2
        remaining = list((await session.scalars(select(BenchmarkCheckpointModel.sweep_id))).all())
        assert set(remaining) == {sweeps[0].id, sweeps[1].id}


async def test_fresh_sweep_worker_restores_database_without_data_dir(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setattr(settings, "data_dir", "/proc/unavailable")
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    factory = session_factory(sessions)
    async with sessions.begin() as session:
        sweep = await SQLAlchemyBenchmarkSweepRepository(session).create(
            BenchmarkSweep(
                search_space={"top_k": {"values": [2]}},
                objective_weights={"hit_rate": 1},
                top_n_llm=1,
            )
        )
        assert sweep.id is not None

    def new_engine() -> SweepEngine:
        return build_sweep_engine(
            factory, checkpoint_factory=lambda sweep_id: DatabaseBenchmarkCheckpoints(factory, sweep_id)
        )

    first = new_engine()
    monkeypatch.setattr(
        first, "load_questions", AsyncMock(return_value=[{"question": "q", "source_hint": "law.pdf"}])
    )
    monkeypatch.setattr(first, "cache_candidates", AsyncMock(return_value=({}, {}, {})))
    monkeypatch.setattr(first._data, "cache_reranker_scores", AsyncMock(return_value={}))
    monkeypatch.setattr(first, "score_config_cheap", lambda *_args: {"composite_score": 1})
    first_phase = AsyncMock(side_effect=RuntimeError("judge unavailable"))
    monkeypatch.setattr(first, "run_phase_b", first_phase)
    with pytest.raises(RuntimeError, match="judge unavailable"):
        await first.run_sweep(sweep, "judge")
    resumed = new_engine()
    load_questions = AsyncMock(side_effect=AssertionError("dataset repeated"))
    cache_candidates = AsyncMock(side_effect=AssertionError("retrieval repeated"))
    resumed_phase = AsyncMock(return_value=[])
    monkeypatch.setattr(resumed, "load_questions", load_questions)
    monkeypatch.setattr(resumed, "cache_candidates", cache_candidates)
    monkeypatch.setattr(resumed, "run_phase_b", resumed_phase)
    await resumed.run_sweep(sweep, "judge")
    load_questions.assert_not_awaited()
    cache_candidates.assert_not_awaited()
    assert resumed_phase.call_args.args[0] == first_phase.call_args.args[0]


async def test_concurrent_sweeps_with_equal_configs_keep_separate_results(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        first = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus.DONE))
        assert first.id is not None
        second = await repo.create(BenchmarkSweep(status=BenchmarkSweepStatus.DONE))
        assert second.id is not None

    async def save(sweep_id: int, score: float) -> BenchmarkRun:
        async with sessions.begin() as session:
            return await SQLAlchemyBenchmarkRunRepository(session).save_for_sweep(
                BenchmarkRun(
                    sweep_id=sweep_id,
                    config_json={"top_k": 4},
                    summary_metrics={"hit_rate": score},
                )
            )

    one, two = await asyncio.gather(save(first.id, 0.9), save(second.id, 0.4))
    assert one.id != two.id
    assert one.sweep_id == first.id
    assert two.sweep_id == second.id
    assert one.summary_metrics["hit_rate"] == 0.9
    assert two.summary_metrics["hit_rate"] == 0.4
