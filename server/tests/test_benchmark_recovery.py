"""Durable recovery and bounded provider calls, without external services."""

import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import instructor
import pytest
from openai import OpenAI


from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from application.services.benchmark_services import BenchmarkSweepService
from fakes import FakeUnitOfWorkFactory
from infrastructure.benchmark import judge, persistence, runner
from infrastructure.benchmark.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.answer_generators import BenchmarkAnswer


@pytest.fixture(autouse=True)
def individual_judge_mode(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "benchmark_judge_grouped_enabled", False)


def test_invalid_provider_request_is_not_retried():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": "invalid request"}})

    raw = OpenAI(
        api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    client = instructor.from_openai(raw, mode=instructor.Mode.MD_JSON)
    with pytest.raises(Exception, match="invalid request"):
        judge.judge_with_structured_output(client, "score the answer", "test-model")
    assert len(requests) == 1
    assert b'"tools"' not in requests[0].content
    assert json.loads(requests[0].content)["max_tokens"] == judge.JUDGE_OUTPUT_TOKENS
    raw.close()


@pytest.mark.asyncio
async def test_resume_skips_completed_questions_and_invalidates_changed_settings(tmp_path, monkeypatch):
    from infrastructure.benchmark import case_evaluator

    calls = []
    fail = True

    async def evaluate(self, idx, question, run_idx, ctx, **kwargs):
        calls.append(question["question"])
        if fail and idx == 2:
            raise RuntimeError("provider rejected request")
        return {"id": idx, "question": question["question"]}

    monkeypatch.setattr(case_evaluator.BenchmarkCaseEvaluator, "evaluate", evaluate)
    monkeypatch.setattr(persistence, "log_question_result", lambda *a: None)
    monkeypatch.setattr(persistence, "log_summary", lambda *a: None)
    monkeypatch.setattr(persistence, "save_results", lambda *a, **kw: None)
    questions = [{"question": "first"}, {"question": "second"}]
    kwargs = {
        "questions": questions,
        "out_dir": str(tmp_path),
        "top_k": 4,
        "judge_model": "test",
        "max_concurrent": 1,
        "rag_service": object(),
        "resume": True,
    }
    with pytest.raises(RuntimeError, match="provider rejected"):
        await runner.run_benchmark_async(**kwargs)
    fail = False
    await runner.run_benchmark_async(**kwargs)
    assert calls == ["first", "second", "second"]
    kwargs["judge_model"] = "another-model"
    await runner.run_benchmark_async(**kwargs)
    assert calls[-2:] == ["first", "second"]


@pytest.mark.asyncio
async def test_failed_judge_keeps_generated_answer(tmp_path, monkeypatch):
    generator = SimpleNamespace(
        generate=AsyncMock(
            return_value=BenchmarkAnswer(
                answer="answer", context="", retriever_metrics={}, input_tokens=12, output_tokens=3
            )
        )
    )
    judge_call = AsyncMock(side_effect=RuntimeError("provider failed"))
    monkeypatch.setattr(judge, "judge_answer_async", judge_call)
    evaluator = BenchmarkCaseEvaluator(generator, "test")
    path = tmp_path / "stages.json"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="provider failed"):
            await evaluator.evaluate(1, {"question": "test"}, 1, SimpleNamespace(), path)
    generator.generate.assert_awaited_once()
    assert judge_call.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", list(BenchmarkSweepStatus))
async def test_resume_only_accepts_terminal_failed_or_cancelled_sweeps(status):
    from domain.exceptions import ValidationError

    factory = FakeUnitOfWorkFactory()
    entity = BenchmarkSweep(id=1, status=status.value)
    repo = factory._uow.benchmark_sweeps
    repo.get_by_id = AsyncMock(return_value=entity)
    repo.requeue = AsyncMock(
        return_value=status in (BenchmarkSweepStatus.FAILED, BenchmarkSweepStatus.CANCELLED)
    )
    service = BenchmarkSweepService(factory)
    if status in (BenchmarkSweepStatus.FAILED, BenchmarkSweepStatus.CANCELLED):
        resumed = await service.resume(1)
        assert resumed.status == BenchmarkSweepStatus.PENDING.value
    else:
        with pytest.raises(ValidationError):
            await service.resume(1)


@pytest.mark.asyncio
async def test_resume_waits_for_cancelled_worker_to_stop():
    from domain.exceptions import BusinessRuleViolation
    from domain.value_objects.job_status import BackgroundJobStatus

    factory = FakeUnitOfWorkFactory()
    factory._uow.benchmark_sweeps.get_by_id = AsyncMock(
        return_value=BenchmarkSweep(id=1, job_id=2, status=BenchmarkSweepStatus.CANCELLED.value)
    )
    factory._uow.background_jobs.get_by_id = AsyncMock(
        return_value=SimpleNamespace(status=BackgroundJobStatus.RUNNING.value)
    )
    factory._uow.benchmark_sweeps.requeue = AsyncMock()
    with pytest.raises(BusinessRuleViolation, match="still stopping"):
        await BenchmarkSweepService(factory).resume(1)
    factory._uow.benchmark_sweeps.requeue.assert_not_awaited()


def test_judge_usage_includes_validation_retry_responses():
    from infrastructure.benchmark.token_usage import judge_usage, record_judge_usage

    responses = iter(["invalid json", '{"score":8,"reason":"ok"}'])

    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": next(responses)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
        )

    raw = OpenAI(
        api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    client = instructor.from_openai(raw, mode=instructor.Mode.MD_JSON)
    client.on("completion:response", record_judge_usage)
    records = []
    token = judge_usage.set(records)
    try:
        assert judge.judge_with_structured_output(client, "score", "test").score == 8
        assert sum(r["input_tokens"] for r in records) == 200
        assert len(records) == 2
    finally:
        judge_usage.reset(token)
        raw.close()
