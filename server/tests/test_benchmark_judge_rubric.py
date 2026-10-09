"""Judge criteria wiring, resume invalidation and sweep objective regressions."""

import json

import pytest

from domain.value_objects.benchmark_scoring import DEFAULT_OBJECTIVE_WEIGHTS
from infrastructure.benchmark import answer_generators, grouped_judge, runner
from infrastructure.benchmark.case_evaluator import refresh_judge_rubric
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.benchmark.judge import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT
from infrastructure.benchmark.judge_rubric import (
    CORRECTNESS_CRITERIA,
    FAITHFULNESS_CRITERIA,
    JUDGE_RUBRIC_VERSION,
)
from infrastructure.benchmark.sweep_scoring import compute_composite_score
from fakes import FakeUnitOfWorkFactory
from test_grouped_benchmark_judge import JudgeClient, case, generator, requested_metrics, response


def test_both_judge_paths_use_the_same_claim_support_criteria():
    assert FAITHFULNESS_CRITERIA in FAITHFULNESS_PROMPT.format(context="", question="?", answer="a")
    assert grouped_judge.INSTRUCTIONS["faithfulness"] == FAITHFULNESS_CRITERIA
    assert CORRECTNESS_CRITERIA in CORRECTNESS_PROMPT.format(question="?", expected="a", answer="a")
    assert grouped_judge.INSTRUCTIONS["correctness"] == CORRECTNESS_CRITERIA


@pytest.mark.asyncio
async def test_case_140_reference_is_not_sent_as_faithfulness_evidence(monkeypatch):
    payloads = []

    async def create(**kwargs):
        metrics = requested_metrics(kwargs)
        prompt = kwargs["messages"][0]["content"]
        payloads.append((metrics, json.loads(prompt.split("Данные JSON:\n")[1])))
        return response({key: {"score": 0, "reason": "test"} for key in metrics})

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda: JudgeClient(create))
    await grouped_judge.judge_case_grouped(
        {"question": "поле?", "expected_answer": "эталон", "annotations": {"required_facts": ["эталон"]}},
        "эталон",
        "другой фрагмент",
        evidence_context="другой фрагмент",
        model="judge",
    )
    support = next(payload for metrics, payload in payloads if "faithfulness" in metrics)
    assert support["context"] == "другой фрагмент"
    assert "expected_answer" not in support


@pytest.mark.parametrize("version", [None, "old-rubric"])
def test_rubric_change_rejudges_without_regenerating_answers(version):
    generated = {"answer": "saved", "context": "source", "evidence": {"context": "source"}}
    stages = {
        "generated": generated,
        "judge_rubric_version": version,
        "grouped_metrics": {"faithfulness": {"score": 10}},
        "generator_metrics": {"faithfulness": 10},
        "context_metrics": {"context_precision": 8},
        "evidence_judge": {"scores": {"citation_support_score": 10}},
    }
    assert refresh_judge_rubric(stages) is True
    assert stages == {"generated": generated, "judge_rubric_version": JUDGE_RUBRIC_VERSION}
    stages["grouped_metrics"] = {"faithfulness": {"score": 0}}
    assert refresh_judge_rubric(stages) is False
    assert stages["grouped_metrics"]["faithfulness"]["score"] == 0


def test_default_sweep_rewards_correctness_and_prefers_supported_answer_to_lucky_match():
    common = {"hit_rate": 1, "relevancy": 10}
    lucky = compute_composite_score({**common, "correctness": 10, "faithfulness": 0})
    supported = compute_composite_score({**common, "correctness": 7, "faithfulness": 10})
    wrong = compute_composite_score({**common, "correctness": 0, "faithfulness": 10})
    assert DEFAULT_OBJECTIVE_WEIGHTS["correctness"] == 0.3
    assert sum(DEFAULT_OBJECTIVE_WEIGHTS.values()) == pytest.approx(1)
    assert supported > lucky
    assert supported > wrong
    assert lucky == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_resume_rejudges_old_completed_result_and_preserves_generation(tmp_path, monkeypatch):
    calls = []

    async def create(**kwargs):
        metrics = requested_metrics(kwargs)
        calls.append(metrics)
        return response({key: {"score": 8, "reason": "test"} for key in metrics})

    generated = generator()
    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda: JudgeClient(create))
    monkeypatch.setattr(answer_generators, "RagBenchmarkGenerator", lambda *args: generated)
    store = DatabaseBenchmarkCheckpoints(FakeUnitOfWorkFactory(), 42)
    kwargs = {
        "questions": [case()],
        "out_dir": str(tmp_path),
        "top_k": 4,
        "judge_model": "judge",
        "rag_service": object(),
        "resume": True,
        "checkpoint_prefix": "configuration",
        "export_files": False,
        "checkpoints": store,
    }
    await runner.run_benchmark_async(**kwargs)
    for path in ("checkpoints/configuration/1-1.json", "checkpoints/configuration/1-1.stages.json"):
        saved = await store.load(path)
        saved.pop("judge_rubric_version")
        await store.save(path, saved)
    results = await runner.run_benchmark_async(**kwargs)
    assert generated.generate.await_count == 1
    assert len(calls) == 6
    assert results[0]["judge_rubric_version"] == JUDGE_RUBRIC_VERSION
    await runner.run_benchmark_async(**kwargs)
    assert len(calls) == 6  # Current completed result is reused.
