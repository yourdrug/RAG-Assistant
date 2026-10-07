"""Grouped judge contract: independent scores, bounded calls and durable resume."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config import settings
from domain.services.benchmark_judge import judge_coverage
from infrastructure.benchmark import grouped_judge
from infrastructure.benchmark.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence
from fakes import FakeUnitOfWorkFactory


class JudgeClient:
    def __init__(self, respond):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=respond))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def response(values):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(values)))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
    )


def requested_metrics(kwargs):
    prompt = kwargs["messages"][0]["content"]
    return [key for key in grouped_judge.INSTRUCTIONS if f"\n{key}:" in prompt]


def case():
    return {
        "id": 1,
        "question": "question",
        "expected_answer": "reference",
        "annotations": {
            "expected_refusal": False,
            "required_facts": ["fact"],
            "required_conditions": ["condition"],
        },
    }


def generator():
    return SimpleNamespace(
        generate=AsyncMock(
            return_value=BenchmarkAnswer(
                answer="fact [1]",
                context="context",
                retriever_metrics={},
                input_tokens=5,
                output_tokens=2,
                evidence=BenchmarkEvidence(retrieved=[], selected=[], context="context"),
            )
        )
    )


@pytest.mark.asyncio
async def test_three_parallel_calls_with_context_only_twice_and_reference_for_retrieval(monkeypatch):
    calls = []
    active = 0
    peak = 0
    barrier = asyncio.Barrier(3)

    async def create(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append(kwargs)
        await asyncio.wait_for(barrier.wait(), 2)
        active -= 1
        return response({key: {"score": 0, "reason": "valid zero"} for key in requested_metrics(kwargs)})

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda: JudgeClient(create))
    result = await BenchmarkCaseEvaluator(generator(), "judge").evaluate(1, case(), 1, SimpleNamespace())
    assert len(calls) == peak == 3
    payloads = [json.loads(call["messages"][0]["content"].split("Данные JSON:\n")[1]) for call in calls]
    assert sum("context" in payload for payload in payloads) == 2
    retrieval = next(payload for payload in payloads if "context" in payload and "answer" not in payload)
    assert retrieval["expected_answer"] == "reference"
    assert result["generator_metrics"]["correctness"] == 0
    assert result["evidence_metrics"]["refusal_score"] == 0
    assert result["evidence_metrics"]["citation_support_score"] == 0
    assert result["context_metrics"]["context_recall"] == 0
    assert result["judge_calls"] == 3


@pytest.mark.asyncio
async def test_invalid_metric_preserves_siblings_and_retries_only_missing(monkeypatch):
    calls = []

    async def create(**kwargs):
        keys = requested_metrics(kwargs)
        calls.append(keys)
        values = {key: {"score": 8, "reason": "ok"} for key in keys}
        if len(keys) > 1 and "correctness" in keys:
            values["correctness"] = {"score": 20, "reason": "invalid"}
        return response(values)

    snapshots = []

    async def save(values):
        snapshots.append(values)

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda: JudgeClient(create))
    result = await grouped_judge.judge_case_grouped(
        case(), "answer", "context", evidence_context="context", model="judge", callback=save
    )
    assert len(calls) == 4
    assert calls.count(["correctness"]) == 1
    assert any(
        snapshot.get("relevancy", {}).get("score") == 8 and snapshot.get("correctness", {}).get("error")
        for snapshot in snapshots
    )
    assert result["correctness"] == {"score": 8, "reason": "ok"}


@pytest.mark.asyncio
async def test_database_resume_keeps_generated_answer_and_successful_scores_after_restart(
    tmp_path, monkeypatch
):
    factory = FakeUnitOfWorkFactory()
    calls = []
    failing = True

    async def create(**kwargs):
        keys = requested_metrics(kwargs)
        calls.append(keys)
        return response(
            {
                key: {"score": "broken" if failing and key == "correctness" else 8, "reason": "ok"}
                for key in keys
            }
        )

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda: JudgeClient(create))
    first_generator = generator()
    store = DatabaseBenchmarkCheckpoints(factory, 42)
    path = Path("config/case.stages.json")
    first = await BenchmarkCaseEvaluator(first_generator, "judge", store).evaluate(
        1, case(), 1, SimpleNamespace(), path
    )
    assert judge_coverage([first])["judge_error_count"] == 1
    assert first["generator_metrics"]["relevancy"] == 8
    assert not list(tmp_path.iterdir())
    count = len(calls)
    failing = False
    # A fresh evaluator/store has no local state; changed token limits do not invalidate work.
    monkeypatch.setattr(settings, "benchmark_judge_initial_tokens", 4096)
    second_generator = generator()
    second = await BenchmarkCaseEvaluator(
        second_generator, "judge", DatabaseBenchmarkCheckpoints(factory, 42)
    ).evaluate(
        1,
        case(),
        1,
        SimpleNamespace(),
        path,
    )
    second_generator.generate.assert_not_awaited()
    assert calls[count:] == [["correctness"]]
    assert judge_coverage([second])["judge_error_count"] == 0
    assert second["generator_metrics"]["correctness"] == 8
    assert second["judge_calls"] == len(calls)


@pytest.mark.asyncio
async def test_request_concurrency_is_bounded_across_questions(monkeypatch):
    monkeypatch.setattr(settings, "benchmark_judge_max_concurrent", 2)
    active = 0
    peak = 0

    async def create(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return response({key: {"score": 9, "reason": "ok"} for key in requested_metrics(kwargs)})

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda: JudgeClient(create))
    await asyncio.gather(
        *(
            grouped_judge.judge_case_grouped(case(), "a", "c", evidence_context="c", model="j")
            for _ in range(4)
        )
    )
    assert peak == 2


@pytest.mark.asyncio
async def test_all_config_results_saved_and_resume_updates_existing_rows():
    from domain.entities.benchmark_sweep import BenchmarkSweep
    from infrastructure.worker.sweep import save_sweep_results

    factory = FakeUnitOfWorkFactory()
    repo = factory._uow.benchmark_runs
    factory._uow.benchmark_sweeps.set_best_run = AsyncMock()
    results = [
        {
            "config": {"top_k": 1},
            "composite_score": 0.8,
            "llm_evaluated": True,
            "full_metrics": {"results": [{"answer": "a"}]},
        },
        {
            "config": {"top_k": 2},
            "composite_score": None,
            "llm_evaluated": True,
            "full_metrics": {"judge_error_count": 1, "results": [{"answer": "b"}]},
        },
    ]
    winner = await save_sweep_results(factory, BenchmarkSweep(), 42, results)
    assert winner is not None
    assert len(repo.runs) == 2
    assert any(run.summary_metrics["judge_error_count"] == 1 for run in repo.runs.values())
    results[1]["full_metrics"]["judge_error_count"] = 0
    await save_sweep_results(factory, BenchmarkSweep(), 42, results)
    assert len(repo.runs) == 2
    assert all(run.summary_metrics["judge_error_count"] != 1 for run in repo.runs.values())


@pytest.mark.asyncio
async def test_cancel_interrupts_inflight_evaluation_and_awaits_cleanup():
    from domain.entities.benchmark_sweep import BenchmarkSweep
    from infrastructure.benchmark.sweep_engine import SweepCancelled
    from infrastructure.worker.sweep import run_cancellable_sweep

    stopped = asyncio.Event()

    async def evaluate(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with pytest.raises(SweepCancelled):
        await asyncio.wait_for(
            run_cancellable_sweep(
                SimpleNamespace(run_sweep=evaluate),
                BenchmarkSweep(),
                AsyncMock(),
                AsyncMock(return_value=True),
            ),
            2,
        )
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_resume_24_of_30_retries_six_scores_without_regenerating_or_writing_reports(
    tmp_path, monkeypatch
):
    from infrastructure.benchmark import runner
    from infrastructure.benchmark.metrics import compute_summary_metrics

    factory = FakeUnitOfWorkFactory()
    generated = generator()
    calls = []
    failing = True

    async def create(**kwargs):
        keys = requested_metrics(kwargs)
        payload = json.loads(kwargs["messages"][0]["content"].split("Данные JSON:\n")[1])
        calls.append((payload["question"], keys))
        return response(
            {
                key: {
                    "score": "broken"
                    if failing and key == "correctness" and int(payload["question"]) >= 24
                    else 8,
                    "reason": "ok",
                }
                for key in keys
            }
        )

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda: JudgeClient(create))
    from infrastructure.benchmark import answer_generators

    monkeypatch.setattr(answer_generators, "RagBenchmarkGenerator", lambda *args: generated)
    questions = [{**case(), "id": idx, "question": str(idx)} for idx in range(30)]
    kwargs = {
        "questions": questions,
        "out_dir": str(tmp_path),
        "top_k": 4,
        "judge_model": "judge",
        "max_concurrent": 3,
        "rag_service": object(),
        "resume": True,
        "checkpoint_prefix": "configuration",
        "export_files": False,
    }
    first = await runner.run_benchmark_async(**kwargs, checkpoints=DatabaseBenchmarkCheckpoints(factory, 42))
    summary = compute_summary_metrics(first)
    assert summary["judge_evaluated_count"] == 24
    assert summary["judge_error_count"] == 6
    assert generated.generate.await_count == 30
    failing = False
    first_call_count = len(calls)
    second = await runner.run_benchmark_async(**kwargs, checkpoints=DatabaseBenchmarkCheckpoints(factory, 42))
    assert generated.generate.await_count == 30
    assert len(calls) - first_call_count == 6
    assert all(keys == ["correctness"] for _, keys in calls[first_call_count:])
    assert compute_summary_metrics(second)["judge_evaluated_count"] == 30
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_legacy_stages_import_preserves_valid_zero_without_local_writes(tmp_path):
    from infrastructure.benchmark.checkpoint import import_legacy_configuration, write_checkpoint

    root = tmp_path / "old-config"
    write_checkpoint(
        root / "checkpoints" / "old-identity" / "1-1.stages.json",
        {"generated": {"answer": "original"}, "generator_metrics": {"faithfulness": 0}},
    )
    factory = FakeUnitOfWorkFactory()
    store = DatabaseBenchmarkCheckpoints(factory, 42)
    await import_legacy_configuration(store, root, "new-identity")
    saved = await store.load("checkpoints/new-identity/1-1.stages.json")
    assert saved["generator_metrics"]["faithfulness"] == 0
    assert saved["generated"]["answer"] == "original"


@pytest.mark.asyncio
async def test_service_returns_current_results_without_consulting_report_files(tmp_path):
    from application.services.benchmark_orchestrator import BenchmarkService

    result = {
        "id": 1,
        "question": "q",
        "answer": "a",
        "latency_sec": 1,
        "generator_metrics": {"faithfulness": 8, "relevancy": 7, "correctness": None},
        "retriever_metrics": {"hit_rate": None, "mrr": None, "avg_similarity": 0},
    }
    (tmp_path / "benchmark_old.json").write_text("invalid old report")
    run = AsyncMock(return_value=[result])
    summary = await BenchmarkService(runner=SimpleNamespace(run=run)).run(
        [{"question": "q"}],
        str(tmp_path),
        4,
        "judge",
    )
    assert summary["avg_faithfulness"] == 8
    assert summary["results"][0]["answer"] == "a"
    assert run.await_args.kwargs["export_files"] is False


@pytest.mark.asyncio
async def test_result_detail_api_serializes_per_question_list():
    from application.services.benchmark_result_service import BenchmarkResultService
    from domain.entities.benchmark_run import BenchmarkRun
    from presentation.api.routes.benchmark import get_benchmark_result

    factory = FakeUnitOfWorkFactory()
    factory._uow.benchmark_runs.get_by_id = AsyncMock(
        return_value=BenchmarkRun(
            id=7,
            per_question_results=[{"id": 1, "answer": "a", "faithfulness": 0}],
        )
    )
    result = await get_benchmark_result(7, admin=SimpleNamespace(), service=BenchmarkResultService(factory))
    assert result.model_dump()["per_question_results"] == [{"id": 1, "answer": "a", "faithfulness": 0}]


@pytest.mark.asyncio
async def test_legacy_single_finalist_uses_latest_attempt_after_operational_setting_change(tmp_path):
    import os
    from infrastructure.benchmark.checkpoint import import_legacy_configuration, write_checkpoint

    root = tmp_path / "sweep"
    write_checkpoint(root / "phase-a-old.json", [{"config": {"top_k": 4}}])
    os.utime(root / "phase-a-old.json", (10, 10))
    for name, timestamp in (("older", 20), ("latest", 30)):
        write_checkpoint(root / name / "benchmark_report.json", [])
        os.utime(root / name / "benchmark_report.json", (timestamp, timestamp))
        write_checkpoint(root / name / "checkpoints" / "old-namespace" / "1-1.stages.json", {"answer": name})
    factory = FakeUnitOfWorkFactory()
    store = DatabaseBenchmarkCheckpoints(factory, 42, legacy_root=root)
    assert await store.load("phase-a-new-hash.json") == [{"config": {"top_k": 4}}]
    await import_legacy_configuration(store, root / "new-hash", "new", single_config_root=root)
    assert await store.load("checkpoints/new/1-1.stages.json") == {"answer": "latest"}


@pytest.mark.asyncio
async def test_failure_does_not_overwrite_acknowledged_cancellation(monkeypatch):
    from domain.entities.benchmark_sweep import BenchmarkSweep
    from domain.value_objects.sweep_status import BenchmarkSweepStatus
    from infrastructure.worker import sweep

    factory = FakeUnitOfWorkFactory()
    factory._uow.benchmark_sweeps.get_by_id = AsyncMock(
        return_value=BenchmarkSweep(
            id=42,
            status=BenchmarkSweepStatus.CANCELLED.value,
        )
    )
    factory._uow.benchmark_sweeps.update_status = AsyncMock()
    monkeypatch.setattr(sweep, "publish_sweep_event", AsyncMock())
    assert await sweep.record_sweep_failure(factory, 42, "judge failed") is True
    factory._uow.benchmark_sweeps.update_status.assert_not_awaited()
    assert sweep.publish_sweep_event.await_args.args[1]["cancelled"] is True
