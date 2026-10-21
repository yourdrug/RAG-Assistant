"""PostgreSQL run identity, compatible baselines, export and checkpoint retention."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.benchmark import BenchmarkExportService
from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweep
from infrastructure.database.models import BenchmarkRunModel, BenchmarkCheckpointModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_job_and_import_retry_are_idempotent_and_do_not_mix_sweeps(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    async def save():
        async with sessions.begin() as session:
            return await SQLAlchemyBenchmarkRunRepository(session).save_once(BenchmarkRun(job_id=55))

    saved = await asyncio.gather(save(), save())
    assert saved[0].id == saved[1].id
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        imported = await repo.save_once(BenchmarkRun(legacy_hash='a' * 64))
        repeated = await repo.save_once(BenchmarkRun(legacy_hash='a' * 64))
        assert imported.id == repeated.id
        assert await session.scalar(select(func.count()).select_from(BenchmarkRunModel)) == 2


async def test_baseline_rejects_other_dataset_mode_and_corpus(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        base = await repo.create(
            BenchmarkRun(
                dataset='legal',
                llm_evaluated=True,
                config_json={'evaluation_mode': 'full', 'collection_name': 'a'},
            )
        )
        await repo.create(
            BenchmarkRun(
                dataset='other',
                llm_evaluated=True,
                config_json={'evaluation_mode': 'full', 'collection_name': 'a'},
            )
        )
        await repo.create(
            BenchmarkRun(
                dataset='legal',
                llm_evaluated=True,
                config_json={'evaluation_mode': 'fast', 'collection_name': 'a'},
            )
        )
        await repo.create(
            BenchmarkRun(
                dataset='legal',
                llm_evaluated=True,
                config_json={'evaluation_mode': 'full', 'collection_name': 'b'},
            )
        )
        current = await repo.create(
            BenchmarkRun(
                dataset='legal',
                llm_evaluated=True,
                config_json={'evaluation_mode': 'full', 'collection_name': 'a'},
            )
        )
        assert (await repo.get_baseline(current)).id == base.id


async def test_export_pages_real_json_array_and_nullable_legacy_results(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        run = await repo.create(BenchmarkRun(per_question_results=[{'id': n} for n in range(4)]))
        legacy = await repo.create(BenchmarkRun())
        assert await repo.result_page(run.id, offset=1, limit=2) == [{'id': 1}, {'id': 2}]
        assert await repo.result_page(legacy.id, offset=0, limit=1) == []

    class Factory:
        @asynccontextmanager
        async def create(self, **kwargs):
            async with sessions.begin() as session:
                yield SimpleNamespace(benchmark_runs=SQLAlchemyBenchmarkRunRepository(session))

    import json

    data = b''.join([block async for block in BenchmarkExportService(Factory()).export(run.id, 'json')])
    assert json.loads(data)['results'] == [{'id': n} for n in range(4)]
    uploaded = []

    class Storage:
        async def upload_stream(self, key, blocks):
            uploaded.append((key, b''.join([block async for block in blocks])))

    assert await BenchmarkExportService(Factory(), Storage()).save_report(run.id, 'json') is True
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        saved = await repo.get_by_id(run.id)
        assert saved.artifacts['json']['status'] == 'ready'
        assert saved.artifacts['json']['size_bytes'] == len(uploaded[0][1])
        assert saved.per_question_results == [{'id': n} for n in range(4)]


async def test_db_retention_preserves_active_and_recent_failed_checkpoints(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        sweeps = []

        for status in ('running', 'failed', 'cancelled', 'done'):
            sweep = await repo.create(BenchmarkSweep(status=status))
            sweeps.append(sweep)
            await repo.save_checkpoint(sweep.id, 'dataset.json', [])

        # Failed is recent; cancelled is old. Done is immediately disposable.
        await session.execute(
            update(BenchmarkCheckpointModel)
            .where(BenchmarkCheckpointModel.sweep_id.in_([sweeps[0].id, sweeps[2].id]))
            .values(updated_at=datetime.now(UTC) - timedelta(days=2))
        )
        assert await repo.cleanup_checkpoints(datetime.now(UTC) - timedelta(days=1)) == 2
        remaining = list((await session.scalars(select(BenchmarkCheckpointModel.sweep_id))).all())
        assert set(remaining) == {sweeps[0].id, sweeps[1].id}




async def test_fresh_sweep_worker_restores_database_without_data_dir(pg_engine, monkeypatch):
    from unittest.mock import AsyncMock
    from config import settings
    from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
    from infrastructure.benchmark.sweep_engine import SweepEngine

    monkeypatch.setattr(settings, 'data_dir', '/proc/unavailable')
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    class Factory:
        @asynccontextmanager
        async def create(self, **kwargs):
            async with sessions.begin() as session:
                yield SimpleNamespace(benchmark_sweeps=SQLAlchemyBenchmarkSweepRepository(session))

    factory = Factory()
    async with sessions.begin() as session:
        sweep = await SQLAlchemyBenchmarkSweepRepository(session).create(
            BenchmarkSweep(
                search_space={'top_k': {'values': [2]}},
                objective_weights={'hit_rate': 1},
                top_n_llm=1,
            )
        )

    def new_engine():
        return SweepEngine(
            factory, checkpoint_factory=lambda sweep_id, root: DatabaseBenchmarkCheckpoints(factory, sweep_id)
        )

    first = new_engine()
    first.load_questions = AsyncMock(return_value=[{'question': 'q', 'source_hint': 'law.pdf'}])
    first.cache_candidates = AsyncMock(return_value=({}, {}, {}))
    first._data.cache_reranker_scores = AsyncMock(return_value={})
    first.score_config_cheap = lambda *args: {'composite_score': 1}
    first.run_phase_b = AsyncMock(side_effect=RuntimeError('judge unavailable'))
    with pytest.raises(RuntimeError, match='judge unavailable'):
        await first.run_sweep(sweep, 'judge')
    resumed = new_engine()
    resumed.load_questions = AsyncMock(side_effect=AssertionError('dataset repeated'))
    resumed.cache_candidates = AsyncMock(side_effect=AssertionError('retrieval repeated'))
    resumed.run_phase_b = AsyncMock(return_value=[])
    await resumed.run_sweep(sweep, 'judge')
    resumed.load_questions.assert_not_awaited()
    resumed.cache_candidates.assert_not_awaited()
    assert resumed.run_phase_b.call_args.args[0] == first.run_phase_b.call_args.args[0]


async def test_concurrent_sweeps_with_equal_configs_keep_separate_results(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkSweepRepository(session)
        first = await repo.create(BenchmarkSweep(status='done'))
        second = await repo.create(BenchmarkSweep(status='done'))

    async def save(sweep_id, score):
        async with sessions.begin() as session:
            return await SQLAlchemyBenchmarkRunRepository(session).save_for_sweep(
                BenchmarkRun(
                    sweep_id=sweep_id,
                    config_json={'top_k': 4},
                    summary_metrics={'hit_rate': score},
                )
            )

    one, two = await asyncio.gather(save(first.id, 0.9), save(second.id, 0.4))
    assert one.id != two.id
    assert one.sweep_id == first.id
    assert two.sweep_id == second.id
    assert one.summary_metrics['hit_rate'] == 0.9
    assert two.summary_metrics['hit_rate'] == 0.4
