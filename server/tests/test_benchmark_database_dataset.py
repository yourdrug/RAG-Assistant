"""Database-only execution contracts for benchmark and both sweep phases."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from application.services.benchmark_dataset import load_benchmark_questions
from domain.entities.benchmark_question import BenchmarkQuestion
from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.value_objects.benchmark_dataset import BenchmarkDataset
from infrastructure.benchmark.sweep_engine import SweepEngine
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)
from presentation.api.schemas.benchmark import BenchmarkRequest


def dataset_factory(questions=None, error=None):
    repo = SimpleNamespace(list_active=AsyncMock(return_value=questions or [], side_effect=error))
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=SimpleNamespace(benchmark_questions=repo))
    uow.__aexit__ = AsyncMock(return_value=False)
    factory = SimpleNamespace(create=MagicMock(return_value=uow))
    return factory, repo, uow


@pytest.mark.asyncio
async def test_dataset_preserves_all_ids_and_annotations_without_1000_question_limit():
    cases = [BenchmarkQuestion(id=i, question=f"Question {i}", dataset="custom") for i in range(1, 1002)]
    cases[0].expected_answer = "Expected"
    cases[0].source_hint = "rules.pdf"
    cases[0].annotations = {"expected_refusal": True}
    cases[0].tags = ["regression"]
    factory, repo, uow = dataset_factory(cases)
    loaded = await load_benchmark_questions(factory, "custom")
    repo.list_active.assert_awaited_once_with(dataset="custom")
    assert len(loaded) == 1001
    assert loaded[0] == {
        "id": 1,
        "question": "Question 1",
        "expected_answer": "Expected",
        "source_hint": "rules.pdf",
        "annotations": {"expected_refusal": True},
        "tags": ["regression"],
    }
    assert loaded[-1]["id"] == 1001
    uow.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_dataset_fails_without_creating_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory, _, _ = dataset_factory()
    with pytest.raises(ValueError, match="No active benchmark questions.*custom"):
        await load_benchmark_questions(factory, "custom")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_db_error_propagates_without_file_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory, _, _ = dataset_factory(error=RuntimeError("database unavailable"))
    with pytest.raises(RuntimeError, match="database unavailable"):
        await load_benchmark_questions(factory, BenchmarkDataset.MAIN.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_repository_selects_only_active_dataset_without_pagination():
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session = SimpleNamespace(execute=AsyncMock(return_value=result))
    await SQLAlchemyBenchmarkQuestionRepository(session).list_active(dataset="custom")
    stmt = session.execute.call_args.args[0]
    sql = str(stmt.compile())
    assert "benchmark_questions.is_active IS true" in sql
    assert "benchmark_questions.dataset =" in sql
    assert "ORDER BY benchmark_questions.id" in sql
    assert "LIMIT" not in sql
    assert stmt.compile().params == {"dataset_1": "custom"}


@pytest.mark.asyncio
async def test_phase_b_uses_loaded_dataset_including_cases_without_source_hints(monkeypatch):
    labelled = BenchmarkQuestion(id=10, question="Labelled", source_hint="rules.pdf", dataset="custom")
    unlabelled = BenchmarkQuestion(
        id=20, question="Refusal", annotations={"expected_refusal": True}, dataset="custom"
    )
    factory, repo, _ = dataset_factory([labelled, unlabelled])
    benchmark = SimpleNamespace(run=AsyncMock(return_value={"hit_rate": 1, "avg_faithfulness": 8, "avg_relevancy": 7}))
    engine = SweepEngine(factory, benchmark_service=benchmark)
    cache = AsyncMock(return_value=({}, {}, {}))
    monkeypatch.setattr(engine, "_cache_candidates", cache)

    def score(config, *args):
        # Simulate dataset editing between phases. B must use the original snapshot.
        repo.list_active.return_value = [BenchmarkQuestion(id=30, question="Changed")]
        return {"composite_score": config["top_k"]}

    monkeypatch.setattr(engine, "_score_config_cheap", score)
    results = await engine.run_sweep(
        BenchmarkSweep(dataset="custom", search_space={"top_k": {"values": [2, 5]}}, top_n_llm=2),
        judge_model="judge",
    )
    repo.list_active.assert_awaited_once_with(dataset="custom")
    assert [q["id"] for q in cache.call_args.args[0]] == [10]
    assert benchmark.run.await_count == 2
    snapshots = [call.kwargs["questions"] for call in benchmark.run.await_args_list]
    assert snapshots[0] is snapshots[1]
    assert [q["id"] for q in snapshots[0]] == [10, 20]
    assert snapshots[0][1]["annotations"] == {"expected_refusal": True}
    assert all(r["llm_evaluated"] for r in results)


def test_benchmark_request_selects_dataset_and_rejects_file_paths():
    assert BenchmarkRequest().dataset == BenchmarkDataset.MAIN.value
    assert BenchmarkRequest(dataset="custom").dataset == "custom"
    with pytest.raises(ValidationError):
        BenchmarkRequest(questions_path="removed.json")


@pytest.mark.asyncio
async def test_worker_loads_selected_dataset_before_evaluation(monkeypatch):
    from infrastructure.worker import tasks

    factory, repo, _ = dataset_factory([BenchmarkQuestion(id=72, question="Database case")])
    ctx = {
        "container": SimpleNamespace(infrastructure=SimpleNamespace(db=SimpleNamespace(uow_factory=factory)))
    }
    run = AsyncMock()
    monkeypatch.setattr(tasks, "run_benchmark_async", run)

    async def tracked(factory_arg, job_id, action, **kwargs):
        assert factory_arg is factory
        assert job_id == 8
        await action()

    monkeypatch.setattr(tasks, "_run_tracked_job", tracked)
    await tasks.run_benchmark(
        ctx, dataset="custom", out_dir="results", top_k=7, judge_model="judge", job_id=8
    )
    repo.list_active.assert_awaited_once_with(dataset="custom")
    assert run.call_args.kwargs["questions"][0]["id"] == 72
    assert run.call_args.kwargs["top_k"] == 7


@pytest.mark.asyncio
async def test_queue_serializes_dataset_and_deduplicates_per_dataset(monkeypatch):
    from infrastructure.worker import queue

    enqueue = AsyncMock()
    monkeypatch.setattr(queue, "_enqueue_arq", enqueue)
    for dataset in [BenchmarkDataset.MAIN.value, "custom"]:
        await queue.enqueue_benchmark(
            dataset=dataset, out_dir="results", top_k=7, judge_model="judge", job_id=8
        )
    calls = enqueue.await_args_list
    assert [c.kwargs["dataset"] for c in calls] == [BenchmarkDataset.MAIN.value, "custom"]
    assert calls[0].kwargs["arq_job_id"] != calls[1].kwargs["arq_job_id"]
    assert all("questions_path" not in c.kwargs for c in calls)
