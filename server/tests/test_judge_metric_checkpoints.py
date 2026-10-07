"""Independent paid judge work survives errors, interruption and resume."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


from application.services.benchmark_orchestrator import compute_summary_from_results
from domain.services.benchmark_judge import judge_coverage
from domain.value_objects.sweep_evaluation_mode import SweepEvaluationMode
from infrastructure.benchmark import judge, metrics, evidence_judge
from infrastructure.benchmark.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.sweep_full_evaluation import SweepFullEvaluator
from infrastructure.benchmark.sweep_settings import LiveSweepSettings
from infrastructure.ml.clients.llm_schemas import JudgeScore


@pytest.fixture(autouse=True)
def individual_judge_mode(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "benchmark_judge_grouped_enabled", False)


@pytest.mark.asyncio
async def test_resume_only_missing_generator_context_and_evidence_scores(tmp_path, monkeypatch):
    generator = SimpleNamespace(
        generate=AsyncMock(
            return_value=BenchmarkAnswer(
                answer="answer", context="context", retriever_metrics={}, input_tokens=0, output_tokens=0
            )
        )
    )
    calls = []
    failing = True

    def score(client, prompt, model):
        calls.append(prompt)
        # Fail a different independent metric in each scoring group.
        if failing and any(
            marker in prompt
            for marker in (
                "RELEVANCY",
                "CONTEXT RECALL",
                "expected_refusal",
            )
        ):
            raise RuntimeError("judge truncated again")
        return JudgeScore(score=0, reason="valid zero")

    monkeypatch.setattr(judge, "get_judge_client", lambda *args: object())
    monkeypatch.setattr(evidence_judge, "get_judge_client", lambda *args: object())
    monkeypatch.setattr(judge, "judge_with_structured_output", score)
    monkeypatch.setattr(evidence_judge, "judge_with_structured_output", score)
    evaluator = BenchmarkCaseEvaluator(generator, "judge")
    question = {"question": "q", "expected_answer": "a", "annotations": {"expected_refusal": True}}
    path = tmp_path / "stage.json"
    first = await evaluator.evaluate(1, question, 1, SimpleNamespace(), path)
    assert first["generator_metrics"]["faithfulness"] == 0
    assert first["generator_metrics"]["correctness"] == 0
    assert first["generator_metrics"]["relevancy"] is None
    assert first["context_metrics"]["context_precision"] == 0
    assert first["context_metrics"]["context_recall"] is None
    assert first["evidence_metrics"]["refusal_score"] is None
    saved = json.loads(path.read_text())
    assert saved["generator_metrics"]["faithfulness"] == 0
    assert saved["context_metrics"]["context_precision"] == 0
    assert saved["evidence_judge"]["details"]["refusal_score"]["error"]
    first_calls = len(calls)
    failing = False
    second = await evaluator.evaluate(1, question, 1, SimpleNamespace(), path)
    generator.generate.assert_awaited_once()
    assert len(calls) - first_calls == 3
    assert judge_coverage([second])["judge_error_count"] == 0
    assert second["generator_metrics"]["relevancy"] == 0
    assert "relevancy_error" not in second["generator_metrics"]
    assert second["context_metrics"]["context_recall"] == 0
    assert "error" not in second["evidence_diagnostics"]["judge"]["refusal_score"]


@pytest.mark.asyncio
async def test_success_checkpoint_is_written_before_next_metric(tmp_path, monkeypatch):
    path = tmp_path / "stage.json"
    generator = SimpleNamespace(
        generate=AsyncMock(
            return_value=BenchmarkAnswer(
                answer="answer", context="", retriever_metrics={}, input_tokens=0, output_tokens=0
            )
        )
    )
    calls = 0

    def score(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert json.loads(path.read_text())["generator_metrics"]["faithfulness"] == 8
            raise asyncio.CancelledError()
        return JudgeScore(score=8, reason="ok")

    monkeypatch.setattr(judge, "get_judge_client", lambda *args: object())
    monkeypatch.setattr(judge, "judge_with_structured_output", score)
    with pytest.raises(asyncio.CancelledError):
        await BenchmarkCaseEvaluator(generator, "judge").evaluate(
            1, {"question": "q"}, 1, SimpleNamespace(), path
        )
    assert json.loads(path.read_text())["generator_metrics"]["faithfulness"] == 8


def test_missing_judge_scores_are_excluded_from_averages_but_counted_as_missing():
    results = []
    for value in (8, None):
        results.append(
            {
                "id": "q",
                "question": "q",
                "answer": "a",
                "latency_sec": 1,
                "generator_metrics": {
                    "faithfulness": value,
                    "relevancy": 9,
                    "correctness": None,
                    **({"faithfulness_error": "truncated"} if value is None else {}),
                },
                "retriever_metrics": {"hit_rate": None, "mrr": None, "avg_similarity": 0},
            }
        )
    for summarize in (metrics.compute_summary_metrics, compute_summary_from_results):
        summary = summarize(results)
        assert summary["avg_faithfulness"] == 8
        assert summary["faithfulness_evaluated_count"] == 1
        assert summary["faithfulness_expected_count"] == 2
        assert summary["judge_evaluated_count"] == 1
        assert summary["judge_error_count"] == 1


@pytest.mark.asyncio
async def test_incomplete_config_cannot_win_and_does_not_stop_remaining_configs():
    full = AsyncMock(
        side_effect=[
            {"avg_faithfulness": 10, "faithfulness_evaluated_count": 1, "faithfulness_expected_count": 2},
            {"avg_faithfulness": 7, "faithfulness_evaluated_count": 2, "faithfulness_expected_count": 2},
        ]
    )
    results = await SweepFullEvaluator(LiveSweepSettings()).evaluate(
        [{"config": {"top_k": k}, "composite_score": 1} for k in (1, 2)],
        2,
        "judge",
        [],
        {"faithfulness": 1},
        None,
        full,
        evaluation_mode=SweepEvaluationMode.FULL,
    )
    assert full.await_count == 2
    assert results[0]["config"]["top_k"] == 2
    assert results[0]["composite_score"] == 0.7
    assert results[1]["composite_score"] is None
    assert results[1]["evaluation_complete"] is False


@pytest.mark.asyncio
async def test_worker_retains_partial_sweep_for_resume_and_reports_counts(monkeypatch):
    from infrastructure.worker import sweep
    from test_worker_sweep_lifecycle import SweepFactory, sweep_context

    factory = SweepFactory()
    results = [
        {
            "llm_evaluated": True,
            "evaluation_complete": False,
            "composite_score": None,
            "full_metrics": {"judge_evaluated_count": 47, "total_questions": 50, "judge_error_count": 3},
        }
    ]
    monkeypatch.setattr(
        sweep, "create_sweep_engine", lambda infra: SimpleNamespace(run_sweep=AsyncMock(return_value=results))
    )
    persist = AsyncMock(return_value=None)
    monkeypatch.setattr(sweep, "save_sweep_results", persist)
    cleanup = Mock()
    monkeypatch.setattr(sweep, "cleanup_sweep_checkpoints", cleanup)
    monkeypatch.setattr(sweep, "publish_sweep_event", AsyncMock())
    await sweep.run_sweep_task(sweep_context(factory), sweep_id=1, job_id=2)
    persist.assert_awaited_once()
    assert "Оценено 47 из 50, ошибок 3" in factory.jobs.mark_failed.await_args.args[1]
    cleanup.assert_called_once_with(sweep.settings.data_dir, 1, successful=False)
    factory.jobs.mark_done.assert_not_called()


@pytest.mark.asyncio
async def test_worker_never_persists_incomplete_config_as_winner():
    from infrastructure.worker.sweep import save_sweep_results
    from domain.entities.benchmark_sweep import BenchmarkSweep
    from fakes import FakeUnitOfWorkFactory

    factory = FakeUnitOfWorkFactory()
    factory._uow.benchmark_runs.create = AsyncMock()
    factory._uow.benchmark_sweeps.set_best_run = AsyncMock()
    winner = await save_sweep_results(
        factory, BenchmarkSweep(), 1, [{"config": {}, "composite_score": None, "llm_evaluated": True}]
    )
    assert winner is None
    factory._uow.benchmark_runs.create.assert_awaited_once()
    factory._uow.benchmark_sweeps.set_best_run.assert_not_awaited()
