"""Grouped judge contract: independent scores, bounded calls and durable resume."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast
from unittest.mock import AsyncMock

import httpx
from openai import RateLimitError
import pytest

from application.benchmark import BenchmarkResultService, BenchmarkService
from application.benchmark.sweep_jobs import SweepJobService
from application.benchmark.sweep_ports import SweepCancelled
from config import settings
from domain.benchmark.judge import judge_coverage
from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from domain.identity.roles import UserKind
from domain.identity.user_context import UserContext
from domain.rag.context import RagContext
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.benchmark.evaluation import answer_generators, case_evaluator, runner
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics
from infrastructure.benchmark.judging import (
    grouped_judge,
    prompts as judge_prompts,
    requests as judge_requests,
)
from infrastructure.benchmark.judging.token_usage import judge_context
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence
from infrastructure.worker.sweep import run_cancellable_sweep
from presentation.api.routes.benchmark import get_benchmark_result
from tests.fakes import FakeChatRAGPort, FakeUnitOfWorkFactory, require_present


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from openai import AsyncOpenAI

    from application.benchmark.dto import BenchmarkRunnerOptions
    from application.benchmark.sweep import SweepEngine
    from application.chat.ports import ChatRAGPort
    from presentation.api.schemas.auth import CurrentUser


@dataclass
class JudgeCompletions:
    create: Callable[..., Awaitable[SimpleNamespace]]


@dataclass
class JudgeChat:
    completions: JudgeCompletions


class JudgeClient:
    def __init__(self, respond: Callable[..., Awaitable[SimpleNamespace]]) -> None:
        self.chat = JudgeChat(completions=JudgeCompletions(create=respond))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


def response(values: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(values)))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
    )


def requested_metrics(kwargs: Mapping[str, Any]) -> list[str]:
    prompt = kwargs["messages"][0]["content"]
    return [key for key in judge_prompts.INSTRUCTIONS if f"\n{key}:" in prompt]


def case() -> dict[str, Any]:
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


@dataclass
class FakeBenchmarkGenerator:
    generate: AsyncMock


def generator() -> FakeBenchmarkGenerator:
    return FakeBenchmarkGenerator(
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
async def test_three_parallel_calls_with_context_only_twice_and_reference_for_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    active = 0
    peak = 0
    barrier = asyncio.Barrier(3)

    async def create(**kwargs: Any) -> Any:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append(kwargs)
        await asyncio.wait_for(barrier.wait(), 2)
        active -= 1
        return response({key: {"score": 0, "reason": "valid zero"} for key in requested_metrics(kwargs)})

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda **_kwargs: JudgeClient(create))
    result = await BenchmarkCaseEvaluator(generator(), "judge").evaluate(
        1, case(), 1, RagContext(UserContext(user_id=1, user_kind=UserKind.INTERNAL))
    )
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
async def test_invalid_metric_preserves_siblings_and_retries_only_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def create(**kwargs: Any) -> Any:
        keys = requested_metrics(kwargs)
        calls.append(keys)
        values = {key: {"score": 8, "reason": "ok"} for key in keys}

        if len(keys) > 1 and "correctness" in keys:
            values["correctness"] = {"score": 20, "reason": "invalid"}

        return response(values)

    snapshots = []

    async def save(values: dict[str, Any]) -> None:
        snapshots.append(values)

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda **_kwargs: JudgeClient(create))
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = FakeUnitOfWorkFactory()
    calls = []
    failing = True

    async def create(**kwargs: Any) -> Any:
        keys = requested_metrics(kwargs)
        calls.append(keys)
        return response(
            {
                key: {"score": "broken" if failing and key == "correctness" else 8, "reason": "ok"}
                for key in keys
            }
        )

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda **_kwargs: JudgeClient(create))
    first_generator = generator()
    store = DatabaseBenchmarkCheckpoints(factory, 42)
    path = Path("config/case.stages.json")
    first = await BenchmarkCaseEvaluator(first_generator, "judge", store).evaluate(
        1, case(), 1, RagContext(UserContext(user_id=1, user_kind=UserKind.INTERNAL)), path
    )
    assert judge_coverage([first])["judge_error_count"] == 1
    assert first["generator_metrics"]["relevancy"] == 8
    assert not list(tmp_path.iterdir())  # noqa: ASYNC240 - Small local fixture I/O in the test setup.
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
        RagContext(UserContext(user_id=1, user_kind=UserKind.INTERNAL)),
        path,
    )
    second_generator.generate.assert_not_awaited()
    assert calls[count:] == [["correctness"]]
    assert judge_coverage([second])["judge_error_count"] == 0
    assert second["generator_metrics"]["correctness"] == 8
    assert second["judge_calls"] == len(calls)


@pytest.mark.asyncio
async def test_request_concurrency_is_bounded_across_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "benchmark_judge_max_concurrent", 2)
    active = 0
    peak = 0

    async def create(**kwargs: Any) -> Any:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return response({key: {"score": 9, "reason": "ok"} for key in requested_metrics(kwargs)})

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda **_kwargs: JudgeClient(create))
    await asyncio.gather(
        *(
            grouped_judge.judge_case_grouped(case(), "a", "c", evidence_context="c", model="j")
            for _ in range(4)
        )
    )
    assert peak == 2


@pytest.mark.asyncio
async def test_all_config_results_saved_and_resume_updates_existing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    factory = FakeUnitOfWorkFactory()
    repo = factory._uow.fake_benchmark_runs
    monkeypatch.setattr(factory._uow.fake_benchmark_sweeps, "set_best_run", AsyncMock())
    results: list[dict[str, Any]] = [
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
    winner = await SweepJobService(factory, AsyncMock(), AsyncMock()).save_sweep_results(
        BenchmarkSweep(), 42, results
    )
    assert winner is not None
    assert len(repo.runs) == 2
    assert any(run.summary_metrics["judge_error_count"] == 1 for run in repo.runs.values())
    results[1]["full_metrics"]["judge_error_count"] = 0
    await SweepJobService(factory, AsyncMock(), AsyncMock()).save_sweep_results(BenchmarkSweep(), 42, results)
    assert len(repo.runs) == 2
    assert all(run.summary_metrics["judge_error_count"] != 1 for run in repo.runs.values())


@pytest.mark.asyncio
async def test_cancel_interrupts_inflight_evaluation_and_awaits_cleanup() -> None:

    stopped = asyncio.Event()

    async def evaluate(**kwargs: Any) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with pytest.raises(SweepCancelled):
        await asyncio.wait_for(
            run_cancellable_sweep(
                cast("SweepEngine", SimpleNamespace(run_sweep=evaluate)),
                BenchmarkSweep(),
                AsyncMock(),
                AsyncMock(return_value=True),
            ),
            2,
        )
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_resume_24_of_30_retries_six_scores_without_regenerating_or_writing_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    factory = FakeUnitOfWorkFactory()
    generated = generator()
    calls = []
    failing = True

    async def create(**kwargs: Any) -> Any:
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

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda **_kwargs: JudgeClient(create))

    monkeypatch.setattr(answer_generators, "RagBenchmarkGenerator", lambda *_args: generated)
    questions = [{**case(), "id": idx, "question": str(idx)} for idx in range(30)]
    options: BenchmarkRunnerOptions = {
        "max_concurrent": 3,
        "rag_service": FakeChatRAGPort(),
        "resume": True,
        "checkpoint_prefix": "configuration",
        "checkpoints": DatabaseBenchmarkCheckpoints(factory, 42),
    }
    first = await runner.run_benchmark_async(questions, 4, "judge", **options)
    summary = compute_summary_metrics(first)
    assert summary["judge_evaluated_count"] == 24
    assert summary["judge_error_count"] == 6
    assert generated.generate.await_count == 30
    failing = False
    first_call_count = len(calls)
    second = await runner.run_benchmark_async(questions, 4, "judge", **options)
    assert generated.generate.await_count == 30
    assert len(calls) - first_call_count == 6
    assert all(keys == ["correctness"] for _, keys in calls[first_call_count:])
    assert compute_summary_metrics(second)["judge_evaluated_count"] == 30
    assert not list(tmp_path.iterdir())  # noqa: ASYNC240 - Small local fixture I/O in the test setup.


@pytest.mark.asyncio
async def test_service_returns_current_results_without_consulting_report_files(tmp_path: Path) -> None:

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
        4,
        "judge",
    )
    assert summary["avg_faithfulness"] == 8
    assert summary["results"][0]["answer"] == "a"
    assert "export_files" not in require_present(run.await_args).kwargs


@pytest.mark.asyncio
async def test_result_detail_api_serializes_per_question_list(monkeypatch: pytest.MonkeyPatch) -> None:

    factory = FakeUnitOfWorkFactory()
    monkeypatch.setattr(
        factory._uow.fake_benchmark_runs,
        "get_by_id",
        AsyncMock(
            return_value=BenchmarkRun(
                id=7, per_question_results=[{"id": 1, "answer": "a", "faithfulness": 0}]
            )
        ),
    )
    result = await get_benchmark_result(
        run_id=7, _admin=cast("CurrentUser", SimpleNamespace()), service=BenchmarkResultService(factory)
    )
    assert result.model_dump()["per_question_results"] == [{"id": 1, "answer": "a", "faithfulness": 0}]


@pytest.mark.asyncio
async def test_failure_does_not_overwrite_acknowledged_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:

    factory = FakeUnitOfWorkFactory()
    monkeypatch.setattr(
        factory._uow.fake_benchmark_sweeps,
        "get_by_id",
        AsyncMock(return_value=BenchmarkSweep(id=42, status=BenchmarkSweepStatus.CANCELLED)),
    )
    monkeypatch.setattr(factory._uow.fake_benchmark_sweeps, "update_status", AsyncMock())
    publish = AsyncMock()
    assert (
        await SweepJobService(factory, publish, AsyncMock()).record_sweep_failure(42, "judge failed") is True
    )
    cast("AsyncMock", factory._uow.fake_benchmark_sweeps.update_status).assert_not_awaited()
    assert require_present(publish.await_args).args[1]["cancelled"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["empty", "truncated", "no_choices", "invalid_json"])
async def test_bad_provider_responses_retry_with_safe_useful_diagnostics(
    monkeypatch: pytest.MonkeyPatch, kind: object, caplog: pytest.LogCaptureFixture
) -> None:
    calls = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        result = response({"faithfulness": {"score": 9, "reason": "ok"}})

        if len(calls) == 1:
            if kind == "empty":
                result.choices[0].message.content = ""
            elif kind == "truncated":
                result.choices[0].finish_reason = "length"
            elif kind == "no_choices":
                result.choices = []
            else:
                result.choices[0].message.content = "PRIVATE DOCUMENT TEXT"

        return result

    scores: dict[str, Any] = {}
    snapshots = []

    async def save(value: dict[str, Any]) -> None:
        snapshots.append(value)

    await judge_requests.judge_group(
        cast("AsyncOpenAI", JudgeClient(create)), "judge", {}, ["faithfulness"], saved=scores, callback=save
    )
    assert scores["faithfulness"]["score"] == 9
    assert len(calls) == 2
    assert calls[1]["max_tokens"] == settings.benchmark_judge_retry_tokens
    assert snapshots[0]["faithfulness"]["score"] is None
    assert '"content_chars"' in caplog.text
    assert "PRIVATE DOCUMENT TEXT" not in caplog.text
    assert "Expecting value" not in caplog.text


@pytest.mark.asyncio
async def test_group_shape_failure_recovers_metrics_individually_without_rescoring_success() -> None:
    calls = []

    async def create(**kwargs: Any) -> Any:
        keys = requested_metrics(kwargs)
        calls.append(keys)

        if len(keys) > 1:
            return response({"nested": {key: {"score": 8, "reason": "ok"} for key in keys}})

        return response({keys[0]: {"score": 8, "reason": "ok"}})

    scores = {"relevancy": {"score": 0, "reason": "valid zero"}}
    await judge_requests.judge_group(
        cast("AsyncOpenAI", JudgeClient(create)),
        "judge",
        {},
        ["relevancy", "faithfulness", "correctness"],
        saved=scores,
        callback=AsyncMock(),
    )
    assert calls == [
        ["correctness", "faithfulness"],
        ["correctness", "faithfulness"],
        ["faithfulness"],
        ["correctness"],
    ]
    assert scores["relevancy"]["score"] == 0
    assert scores["faithfulness"]["score"] == scores["correctness"]["score"] == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,explicit", [(2, 2), (10, None)])
async def test_questions_really_generate_concurrently_and_return_in_dataset_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int, explicit: int | None
) -> None:

    monkeypatch.setattr(settings, "benchmark_max_concurrent", limit)
    started = asyncio.Barrier(limit)
    active = peak = 0

    async def evaluate(
        self: object, idx: object, question: dict[str, Any], run: object, ctx: object, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.wait_for(started.wait(), 2)
        active -= 1
        return {"id": question["id"]}

    monkeypatch.setattr(answer_generators, "RagBenchmarkGenerator", lambda *_args: object())
    monkeypatch.setattr(case_evaluator.BenchmarkCaseEvaluator, "evaluate", evaluate)
    monkeypatch.setattr(
        "infrastructure.benchmark.reporting.progress.log_question_result", lambda *_args: None
    )
    monkeypatch.setattr("infrastructure.benchmark.reporting.progress.log_summary", lambda *_args: None)
    results = await runner.run_benchmark_async(
        [{"id": i, "question": str(i)} for i in range(limit * 2)],
        4,
        "judge",
        rag_service=cast("ChatRAGPort", object()),
        max_concurrent=explicit,
    )
    assert peak == limit
    assert [result["id"] for result in results] == list(range(limit * 2))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,code",
    [
        ("empty", "empty_content"),
        ("truncated", "output_truncated"),
        ("json", "invalid_json"),
        ("shape", "invalid_metrics"),
        ("no_choices", "no_choices"),
        ("refusal", "provider_refusal"),
    ],
)
async def test_structured_logs_identify_failure_without_private_content(  # noqa: PLR0915 - One scenario verifies the complete transaction/lifecycle.
    kind: str, code: str, caplog: pytest.LogCaptureFixture
) -> None:

    raw = response({"faithfulness": {"score": 8, "reason": "PRIVATE REASON"}})
    raw.id = "completion-id"
    raw._request_id = "provider-id"
    raw.choices[0].finish_reason = "stop"
    raw.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=7)

    if kind == "empty":
        raw.choices[0].message.content = ""
    elif kind == "truncated":
        raw.choices[0].finish_reason = "length"
    elif kind == "json":
        raw.choices[0].message.content = "PRIVATE DOCUMENT"
    elif kind == "shape":
        raw.choices[0].message.content = '{"PRIVATE FIELD": {"score": 8}}'
    elif kind == "no_choices":
        raw.choices = []
    else:
        raw.choices[0].message.refusal = "PRIVATE REFUSAL"

    ctx = judge_context.set({"case_id": "case-id", "question_id": 142, "run": 1})
    try:
        await judge_requests.judge_batch(
            cast("AsyncOpenAI", JudgeClient(AsyncMock(return_value=raw))),
            "judge",
            {"context": "PRIVATE CONTEXT"},
            ["faithfulness"],
            saved={},
            callback=AsyncMock(),
            budget=10,
            attempt=1,
        )
    finally:
        judge_context.reset(ctx)
    event = next(record.benchmark_judge for record in caplog.records if hasattr(record, "benchmark_judge"))
    assert event["error_code"] == code
    assert event["question_id"] == 142
    assert event["case_id"] == "case-id"
    assert event["provider_request_id"] == "provider-id"
    assert event["response_id"] == "completion-id"
    assert event["reasoning_tokens"] == 7
    assert event["queue_wait_sec"] >= 0
    assert event["request_sec"] >= 0
    assert event["elapsed_sec"] >= event["request_sec"]
    assert "PRIVATE" not in caplog.text


@pytest.mark.asyncio
async def test_http_failure_logs_request_id_status_retry_after_without_provider_body(
    caplog: pytest.LogCaptureFixture,
) -> None:

    error = RateLimitError(
        "PRIVATE PROVIDER BODY",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://test"),
            headers={"x-request-id": "provider-id", "retry-after": "2"},
        ),
        body={"private": "PRIVATE DOCUMENT"},
    )
    retryable = await judge_requests.judge_batch(
        cast("AsyncOpenAI", JudgeClient(AsyncMock(side_effect=error))),
        "judge",
        {},
        ["faithfulness"],
        saved={},
        callback=AsyncMock(),
        budget=10,
        attempt=1,
    )
    event = next(record.benchmark_judge for record in caplog.records if hasattr(record, "benchmark_judge"))
    assert retryable
    assert event["http_status"] == 429
    assert event["provider_request_id"] == "provider-id"
    assert event["retry_after"] == "2"
    assert "PRIVATE" not in caplog.text


@pytest.mark.asyncio
async def test_parallel_cases_keep_separate_log_correlation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:

    caplog.set_level(logging.INFO, logger="default")

    async def create(**kwargs: Any) -> SimpleNamespace:
        await asyncio.sleep(0)
        return response({key: {"score": 8, "reason": "ok"} for key in requested_metrics(kwargs)})

    monkeypatch.setattr(grouped_judge, "create_async_judge_client", lambda **_kwargs: JudgeClient(create))
    await asyncio.gather(
        *(
            BenchmarkCaseEvaluator(generator(), "judge").evaluate(
                1,
                {**case(), "id": question_id},
                1,
                RagContext(UserContext(user_id=1, user_kind=UserKind.INTERNAL)),
            )
            for question_id in (11, 22)
        )
    )
    events = [record.benchmark_judge for record in caplog.records if hasattr(record, "benchmark_judge")]
    requests = [event for event in events if event["event"] == "judge_request"]
    assert len(requests) == len({event["request_id"] for event in requests}) == 6
    case_ids = {}

    for question_id in (11, 22):
        matched = [event for event in requests if event["question_id"] == question_id]
        case_ids[question_id] = {event["case_id"] for event in matched}
        assert len(matched) == 3
        assert len(case_ids[question_id]) == 1

    assert case_ids[11].isdisjoint(case_ids[22])
    assert judge_context.get() is None
    assert len([event for event in events if event["event"] == "judge_case_completed"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [None, 2])
async def test_service_passes_runtime_concurrency_through_instead_of_hardcoding_four(
    explicit: int | None,
) -> None:

    run = AsyncMock(return_value=[])
    await BenchmarkService(runner=SimpleNamespace(run=run)).run(
        [{"question": "q"}],
        4,
        "judge",
        max_concurrent=explicit,
    )
    assert require_present(run.await_args).kwargs["max_concurrent"] is explicit
