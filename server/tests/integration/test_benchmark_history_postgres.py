"""PostgreSQL run identity, compatible baselines, export and checkpoint retention."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
import httpx
import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.integration.conftest import upgrade_schema
from tests.integration.uow import session_factory

from application.benchmark import BenchmarkExportService, BenchmarkResultService
from composition.service_providers import build_sweep_engine
from config import settings
from domain.benchmark.evaluation import EVIDENCE_METRICS
from domain.benchmark.policy import JudgeCalibration, JudgeIdentity
from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.database.models import BenchmarkCheckpointModel, BenchmarkRunModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_quality_repository import (
    SQLAlchemyBenchmarkQualityRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)
from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_benchmark_export_service, create_benchmark_result_service
from presentation.api.routes import benchmark as benchmark_routes


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.engine import Connection
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


async def test_run_list_projects_legacy_scores_and_preserves_full_details(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    details = [
        {"answer": "large trace" * 1000, "evidence_metrics": {"answer_condition_coverage": value}}
        for value in (0, 1, 1)
    ]
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        legacy = await repo.create(BenchmarkRun(dataset="legacy", per_question_results=details))
        await repo.create(BenchmarkRun(dataset="other"))

    async with sessions() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        runs = await repo.list_items(dataset="legacy", sort_by="id", sort_order="asc", limit=1)
        assert [run.id for run in runs] == [legacy.id]
        assert runs[0].summary_metrics["avg_answer_condition_coverage"] == 0.6667
        assert runs[0].summary_metrics["answer_condition_coverage_evaluated_count"] == 3
        assert runs[0].per_question_results == [
            {"evidence_metrics": row["evidence_metrics"]} for row in details
        ]
        assert await repo.list_items(dataset="legacy", offset=1) == []

    async with sessions() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        assert legacy.id is not None
        full = await repo.get_by_id(legacy.id)
        assert full is not None
        assert full.per_question_results == details
        assert (
            await session.scalar(
                select(BenchmarkRunModel.summary_metrics).where(BenchmarkRunModel.id == legacy.id)
            )
            == {}
        )


async def test_run_list_skips_details_when_all_evidence_averages_are_saved(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    summary = {f"avg_{metric}": 0.0 for metric in EVIDENCE_METRICS}
    async with sessions.begin() as session:
        await SQLAlchemyBenchmarkRunRepository(session).create(
            BenchmarkRun(summary_metrics=summary, per_question_results=[{"answer": "large trace" * 1000}])
        )
    async with sessions() as session:
        runs = await SQLAlchemyBenchmarkRunRepository(session).list_items()
        assert len(runs) == 1
        assert runs[0].summary_metrics == summary
        assert runs[0].per_question_results is None


def review_results() -> list[dict[str, Any]]:
    trace = "full retrieval trace " * 500
    rows = [
        {
            "id": 41 if n in (2, 28) else n,
            "run": 2 if n == 28 else 1,
            "question": f"Пример {n}",
            "answer": f"Answer {n}",
            "expected_answer": f"Reference {n}",
            "generator_metrics": {"correctness": 10 if n in (2, 28) else 0 if n == 3 else None},
            "evidence": {"context": trace, "completion": "grounded_response"},
            "evidence_metrics": {"answer_fact_coverage": 1 if n == 28 else 0},
        }
        for n in range(1, 30)
    ]
    # A legacy row with only a top-level score also participates in the filter.
    rows[-1].update(correctness=10)
    return rows


async def test_review_api_pages_the_filtered_run_and_preserves_full_results(  # noqa: PLR0915
    pg_engine: AsyncEngine,
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    factory = session_factory(sessions)
    rows = review_results()
    trace = rows[27]["evidence"]["context"]
    async with sessions.begin() as session:
        run = await SQLAlchemyBenchmarkRunRepository(session).create(
            BenchmarkRun(dataset="review", llm_evaluated=True, per_question_results=rows)
        )

    assert run.id is not None
    app = FastAPI()
    app.include_router(benchmark_routes.router)
    app.dependency_overrides[require_admin] = lambda: None
    app.dependency_overrides[create_benchmark_result_service] = lambda: BenchmarkResultService(factory)
    app.dependency_overrides[create_benchmark_export_service] = lambda: BenchmarkExportService(factory)
    path = f"/benchmark/results/{run.id}"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:  # type: ignore[arg-type]
        first = await client.get(f"{path}/questions", params={"score": 10, "limit": 1})
        assert first.status_code == 200, first.text
        page = first.json()
        assert (page["total"], page["matched"], page["scored"]) == (29, 3, 4)
        assert [(item["position"], item["key"]) for item in page["questions"]] == [(2, "41#1")]
        assert "Answer" not in first.text
        assert "Reference" not in first.text
        assert trace not in first.text
        remaining = await client.get(f"{path}/questions", params={"score": 10, "limit": 2, "offset": 1})
        assert [(item["position"], item["key"]) for item in remaining.json()["questions"]] == [
            (28, "41#2"),
            (29, "29#1"),
        ]
        filtered = await client.get(f"{path}/questions", params={"comparison": "lt", "score": 7})
        assert [item["position"] for item in filtered.json()["questions"]] == [3]
        searched = await client.get(f"{path}/questions", params={"query": "пРИМЕР 28", "score": 10})
        assert [item["position"] for item in searched.json()["questions"]] == [28]
        assert (await client.get(f"{path}/questions", params={"score": 11})).status_code == 422
        assert (await client.get(f"{path}/questions/30")).status_code == 404

        detail = (await client.get(f"{path}/questions/28")).json()
        assert detail["answer"] == "Answer 28"
        assert detail["expected_answer"] == "Reference 28"
        assert detail["evidence"]["context"] == trace
        assert detail["question_key"] == "41#2"
        assert detail["outcome"] == "grounded"
        summary = (await client.get(f"{path}/summary")).json()["summary_metrics"]
        assert summary["avg_answer_fact_coverage"] == round(1 / 29, 4)
        assert summary["answer_fact_coverage_evaluated_count"] == 29
        assert sum(summary["diagnosis_counts"].values()) == 29
        export = await client.get(f"{path}/export")
        assert export.json()["results"] == rows


def rebuild_review_projection(connection: Connection) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = connection
    command.downgrade(config, "0006")
    upgrade_schema(connection)


async def test_review_projection_tracks_existing_writers(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    rows = review_results()
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        run = await repo.create(BenchmarkRun(per_question_results=[rows[0]]))
        assert run.id is not None

    async with pg_engine.connect() as connection:
        await connection.run_sync(rebuild_review_projection)

    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkRunRepository(session)
        backfilled = await repo.get_review_run(run.id)
        assert backfilled is not None
        assert backfilled.per_question_results
        assert backfilled.per_question_results[0]["question"] == rows[0]["question"]
        assert "answer" not in backfilled.per_question_results[0]
        # Mixed-version writers still update the projection through the trigger.
        await session.execute(
            update(BenchmarkRunModel)
            .where(BenchmarkRunModel.id == run.id)
            .values(per_question_results=[rows[-1]])
        )
        review = await repo.get_review_run(run.id)
        assert review is not None
        assert review.per_question_results
        assert review.per_question_results[0]["question"] == "Пример 29"
        assert "answer" not in review.per_question_results[0]
        full = await repo.get_by_id(run.id)
        assert full is not None
        assert full.per_question_results == [rows[-1]]


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
    monkeypatch.setattr(first, "score_config_cheap", lambda *_args, **_kwargs: {"composite_score": 1})
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


async def test_calibration_history_survives_run_deletion(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    measured = datetime(2026, 11, 1, tzinfo=UTC)
    async with sessions.begin() as session:
        run = await SQLAlchemyBenchmarkRunRepository(session).save_once(BenchmarkRun(job_id=91))
        repo = SQLAlchemyBenchmarkQualityRepository(session)

        for offset, run_id in enumerate((run.id, None)):
            saved = await repo.add_calibration(
                "support-main",
                JudgeCalibration(
                    run_id=run_id,
                    judge=JudgeIdentity(model="judge-a", rubric_version=f"r{offset}"),
                    metrics={"correctness": {"n": offset}},
                    measured_at=measured + timedelta(days=offset),
                ),
            )
            assert saved.id is not None

        await repo.add_calibration(
            "other-main",
            JudgeCalibration(run_id=None, judge=JudgeIdentity(model=None, rubric_version=None), metrics={}),
        )

    async with sessions.begin() as session:
        await session.execute(delete(BenchmarkRunModel).where(BenchmarkRunModel.id == run.id))

    async with sessions.begin() as session:
        history = await SQLAlchemyBenchmarkQualityRepository(session).list_calibrations("support-main", 10)

    assert [item.judge.rubric_version for item in history] == ["r1", "r0"]
    assert history[1].run_id is None
    assert history[1].metrics == {"correctness": {"n": 0}}
