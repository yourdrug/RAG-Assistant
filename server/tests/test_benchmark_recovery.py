"""Durable recovery and bounded provider calls, without external services."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import httpx
import instructor
from openai import OpenAI
import pytest

from application.benchmark import BenchmarkSweepService
from config import settings
from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from domain.runtime.jobs import BackgroundJobStatus
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.benchmark.evaluation import case_evaluator, runner
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.judging import judge
from infrastructure.benchmark.judging.judge_rubric import JUDGE_RUBRIC_VERSION
from infrastructure.benchmark.judging.token_usage import judge_usage, record_judge_usage
from infrastructure.benchmark.reporting import progress
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from pathlib import Path

    from application.chat.ports import ChatRAGPort
    from domain.rag.context import RagContext
    from infrastructure.ml.clients.managed_sync_instructor import ManagedSyncInstructor


@pytest.fixture(autouse=True)
def individual_judge_mode(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(settings, "benchmark_judge_grouped_enabled", False)


def test_invalid_provider_request_is_not_retried() -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": "invalid request"}})

    raw = OpenAI(
        api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    client = instructor.from_openai(raw, mode=instructor.Mode.MD_JSON)
    with pytest.raises(Exception, match="invalid request"):
        judge.judge_with_structured_output(
            cast("ManagedSyncInstructor", client), "score the answer", "test-model"
        )
    assert len(requests) == 1
    assert b'"tools"' not in requests[0].content
    assert json.loads(requests[0].content)["max_tokens"] == judge.JUDGE_OUTPUT_TOKENS
    raw.close()


@pytest.mark.asyncio
async def test_resume_skips_completed_questions_and_invalidates_changed_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    calls = []
    fail = True

    async def evaluate(
        self: object, idx: Any, question: dict[str, Any], run_idx: object, ctx: object, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append(question["question"])

        if fail and idx == 2:
            msg = "provider rejected request"
            raise RuntimeError(msg)

        return {"id": idx, "question": question["question"], "judge_rubric_version": JUDGE_RUBRIC_VERSION}

    monkeypatch.setattr(case_evaluator.BenchmarkCaseEvaluator, "evaluate", evaluate)
    monkeypatch.setattr(progress, "log_question_result", lambda *_a: None)
    monkeypatch.setattr(progress, "log_summary", lambda *_a: None)
    questions = [{"question": "first"}, {"question": "second"}]
    kwargs = {
        "questions": questions,
        "top_k": 4,
        "judge_model": "test",
        "max_concurrent": 1,
        "rag_service": object(),
        "resume": True,
        "checkpoints": DatabaseBenchmarkCheckpoints(FakeUnitOfWorkFactory(), 1),
    }
    with pytest.raises(RuntimeError, match="provider rejected"):
        await runner.run_benchmark_async(**cast("dict[str, Any]", kwargs))
    fail = False
    await runner.run_benchmark_async(**cast("dict[str, Any]", kwargs))
    assert calls == ["first", "second", "second"]
    kwargs["judge_model"] = "another-model"
    await runner.run_benchmark_async(**cast("dict[str, Any]", kwargs))
    assert calls[-2:] == ["first", "second"]


@pytest.mark.asyncio
async def test_failed_judge_keeps_generated_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = SimpleNamespace(
        generate=AsyncMock(
            return_value=BenchmarkAnswer(
                answer="answer", context="", retriever_metrics={}, input_tokens=12, output_tokens=3
            )
        )
    )
    judge_call = AsyncMock(side_effect=RuntimeError("provider failed"))
    monkeypatch.setattr(judge, "judge_answer_async", judge_call)
    store = DatabaseBenchmarkCheckpoints(FakeUnitOfWorkFactory(), 1)
    evaluator = BenchmarkCaseEvaluator(generator, "test", store)
    path = tmp_path / "stages.json"

    for _ in range(2):
        with pytest.raises(RuntimeError, match="provider failed"):
            await evaluator.evaluate(1, {"question": "test"}, 1, cast("RagContext", SimpleNamespace()), path)

    generator.generate.assert_awaited_once()
    assert judge_call.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", list(BenchmarkSweepStatus))
async def test_resume_only_accepts_terminal_failed_or_cancelled_sweeps(
    status: BenchmarkSweepStatus, monkeypatch: pytest.MonkeyPatch
) -> None:

    factory = FakeUnitOfWorkFactory()
    entity = BenchmarkSweep(id=1, status=status)
    repo = factory._uow.fake_benchmark_sweeps
    monkeypatch.setattr(repo, "get_by_id", AsyncMock(return_value=entity))
    monkeypatch.setattr(
        repo,
        "requeue",
        AsyncMock(return_value=status in (BenchmarkSweepStatus.FAILED, BenchmarkSweepStatus.CANCELLED)),
    )
    service = BenchmarkSweepService(factory)

    if status in (BenchmarkSweepStatus.FAILED, BenchmarkSweepStatus.CANCELLED):
        resumed = await service.resume(1)
        assert resumed.status is BenchmarkSweepStatus.PENDING
    else:
        with pytest.raises(ValidationError):
            await service.resume(1)


@pytest.mark.asyncio
async def test_resume_waits_for_cancelled_worker_to_stop(monkeypatch: pytest.MonkeyPatch) -> None:

    factory = FakeUnitOfWorkFactory()
    monkeypatch.setattr(
        factory._uow.fake_benchmark_sweeps,
        "get_by_id",
        AsyncMock(return_value=BenchmarkSweep(id=1, job_id=2, status=BenchmarkSweepStatus.CANCELLED)),
    )
    monkeypatch.setattr(
        factory._uow.fake_background_jobs,
        "get_by_id",
        AsyncMock(return_value=SimpleNamespace(status=BackgroundJobStatus.RUNNING.value)),
    )
    monkeypatch.setattr(factory._uow.fake_benchmark_sweeps, "requeue", AsyncMock())
    with pytest.raises(BusinessRuleViolationError, match="still stopping"):
        await BenchmarkSweepService(factory).resume(1)
    cast("AsyncMock", factory._uow.fake_benchmark_sweeps.requeue).assert_not_awaited()


def test_judge_usage_includes_validation_retry_responses() -> None:

    responses = iter(["invalid json", '{"score":8,"reason":"ok"}'])

    def respond(request: httpx.Request) -> httpx.Response:
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
    records: list[Any] = []
    token = judge_usage.set(records)
    try:
        assert (
            judge.judge_with_structured_output(cast("ManagedSyncInstructor", client), "score", "test").score
            == 8
        )
        assert sum(r["input_tokens"] for r in records) == 200
        assert len(records) == 2
    finally:
        judge_usage.reset(token)
        raw.close()


@pytest.mark.asyncio
async def test_resume_requires_database_store_before_starting_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="database checkpoint store"):
        await runner.run_benchmark_async(
            [{"question": "q"}], 4, "judge", rag_service=cast("ChatRAGPort", object()), resume=True
        )
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.


@pytest.mark.asyncio
async def test_stage_path_requires_database_store_without_file_fallback(tmp_path: Path) -> None:
    generator = SimpleNamespace(generate=AsyncMock())
    with pytest.raises(RuntimeError, match="database checkpoint store"):
        await BenchmarkCaseEvaluator(generator, "judge").evaluate(
            1, {"question": "q"}, 1, cast("RagContext", SimpleNamespace()), tmp_path / "stages.json"
        )
    generator.generate.assert_not_awaited()
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.
