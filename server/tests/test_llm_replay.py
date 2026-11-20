"""Benchmark LLM replay: only identical requests are reused, and only inside a benchmark scope."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import traceback
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import SecretStr
import pytest
from sqlalchemy.exc import DBAPIError

from application.benchmark import compute_summary_from_results
from application.benchmark.dto import SweepCreateDTO
from application.benchmark.export_service import encode_csv
from application.benchmark.jobs import BenchmarkJobService
from application.benchmark.services import BenchmarkSweepService
from application.benchmark.sweep import SweepRun
from composition.service_providers import build_sweep_engine
from domain.benchmark.llm_replay import LLMReplayMode, LLMReplayStats
from infrastructure.ml.replay.recording import (
    RecordedResponse,
    mark_replayed,
    request_key,
    response_usage,
)
from infrastructure.ml.replay.session import llm_replay_scope
from infrastructure.ml.replay.store import PostgresReplayStore, ReplayStoreError
from infrastructure.ml.replay.transport import (
    LLMReplayMissError,
    ReplayAsyncTransport,
    llm_async_http_client,
    wrap_async_client,
)
from infrastructure.ml.usage_capture import (
    UsageCapturingCompletions,
    active_llm_usage,
    active_stream_response,
    record_model_response,
    summarize_usage,
)
from infrastructure.worker.registry import BenchmarkPayload
from presentation.api.helpers import sweep_create_to_dto, sweep_to_response
from presentation.api.schemas.benchmark import SweepCreateRequest
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


URL = "http://llm.test/v1/chat/completions"
USAGE = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ответ"}, "finish_reason": "stop"}],
    "usage": USAGE,
}


class MemoryStore:
    def __init__(self) -> None:
        self.entries: dict[str, RecordedResponse] = {}
        self.saves = 0

    async def load(self, key: str) -> RecordedResponse | None:
        return self.entries.get(key)

    async def save(self, key: str, model: str | None, response: RecordedResponse) -> None:
        del model
        self.saves += 1
        self.entries[key] = response


class Provider:
    """Counts paid calls and answers each with a chat completion."""

    def __init__(self, status_code: int = 200) -> None:
        self.calls = 0
        self.status_code = status_code
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.headers.append(request.headers)
        return httpx.Response(self.status_code, json=COMPLETION)


def async_client(provider: Provider) -> httpx.AsyncClient:
    return wrap_async_client(httpx.AsyncClient(transport=httpx.MockTransport(provider)))


def body(prompt: str = "вопрос", **extra: object) -> dict[str, object]:
    return {"model": "m", "temperature": 0, "messages": [{"role": "user", "content": prompt}], **extra}


@pytest.mark.asyncio
async def test_calls_outside_a_benchmark_scope_always_reach_the_provider() -> None:
    provider, store = Provider(), MemoryStore()
    client = async_client(provider)

    for _ in range(2):
        response = await client.post(URL, json=body())
        assert response.json()["choices"][0]["message"]["content"] == "ответ"

    assert provider.calls == 2
    assert store.saves == 0


@pytest.mark.asyncio
async def test_record_reuses_an_identical_request_regardless_of_headers_and_key_order() -> None:
    provider, store = Provider(), MemoryStore()
    client = async_client(provider)

    async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
        first = await client.post(URL, json=body(), headers={"Authorization": "Bearer one"})
        reordered = json.dumps(dict(reversed(list(body().items()))), ensure_ascii=False)
        second = await client.post(
            URL,
            content=reordered.encode(),
            headers={"Authorization": "Bearer two", "content-type": "application/json"},
        )

    assert provider.calls == 1
    assert first.json() == COMPLETION
    assert second.json() == {**COMPLETION, "usage": {**USAGE, "replayed": True}}
    assert stats == LLMReplayStats(
        mode=LLMReplayMode.RECORD, hits=1, misses=1, recorded=1, saved_input_tokens=11, saved_output_tokens=3
    )
    assert provider.headers[0]["accept-encoding"] == "identity"


@pytest.mark.asyncio
async def test_changed_prompt_or_parameters_are_paid_again() -> None:
    provider, store = Provider(), MemoryStore()
    client = async_client(provider)

    async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
        await client.post(URL, json=body())
        await client.post(URL, json=body("другой вопрос"))
        await client.post(URL, json=body(temperature=0.7))

    assert provider.calls == 3
    assert stats.hits == 0
    assert stats.recorded == 3


@pytest.mark.asyncio
async def test_failed_responses_are_not_recorded() -> None:
    provider, store = Provider(status_code=429), MemoryStore()
    client = async_client(provider)

    async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
        for _ in range(2):
            assert (await client.post(URL, json=body())).status_code == 429

    assert provider.calls == 2
    assert store.saves == 0
    assert stats.misses == 2


@pytest.mark.asyncio
async def test_replay_only_fails_a_miss_instead_of_paying() -> None:
    provider, store = Provider(), MemoryStore()
    client = async_client(provider)

    async with llm_replay_scope(store, LLMReplayMode.REPLAY_ONLY) as stats:
        with pytest.raises(LLMReplayMissError):
            await client.post(URL, json=body())

    assert provider.calls == 0
    assert stats.misses == 1


@pytest.mark.asyncio
async def test_refresh_pays_and_overwrites_the_recording() -> None:
    provider, store = Provider(), MemoryStore()
    client = async_client(provider)
    key = request_key("POST", URL, json.dumps(body()).encode())
    store.entries[key] = RecordedResponse(200, {"content-type": "application/json"}, b'{"stale": true}')

    async with llm_replay_scope(store, LLMReplayMode.REFRESH) as stats:
        response = await client.post(URL, json=body())

    assert provider.calls == 1
    assert response.json() == COMPLETION
    assert json.loads(store.entries[key].body) == COMPLETION
    assert stats.hits == 0
    assert stats.recorded == 1


@pytest.mark.asyncio
async def test_off_mode_never_touches_the_store() -> None:
    provider, store = Provider(), MemoryStore()
    client = async_client(provider)

    async with llm_replay_scope(store, LLMReplayMode.OFF) as stats:
        await client.post(URL, json=body())
        await client.post(URL, json=body())

    assert provider.calls == 2
    assert store.saves == 0
    assert stats == LLMReplayStats()


@pytest.mark.asyncio
async def test_a_failing_store_degrades_to_paid_calls() -> None:
    provider = Provider()
    store = MemoryStore()
    store.load = AsyncMock(side_effect=OSError("db down"))  # type: ignore[method-assign]
    store.save = AsyncMock(side_effect=OSError("db down"))  # type: ignore[method-assign]
    client = async_client(provider)

    async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
        response = await client.post(URL, json=body())

    assert response.json() == COMPLETION
    assert provider.calls == 1
    assert stats.store_errors == 2
    assert stats.recorded == 0


@pytest.mark.asyncio
async def test_concurrent_chat_request_is_not_replayed_while_a_benchmark_runs() -> None:
    provider, store = Provider(), MemoryStore()
    client = async_client(provider)
    benchmark_started = asyncio.Event()
    chat_done = asyncio.Event()

    async def benchmark() -> LLMReplayStats:
        async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
            await client.post(URL, json=body())
            benchmark_started.set()
            await chat_done.wait()
        return stats

    async def chat() -> None:
        await benchmark_started.wait()
        # The same request a benchmark just recorded: a chat answer must still be generated.
        await client.post(URL, json=body())
        chat_done.set()

    stats, _ = await asyncio.gather(benchmark(), chat())

    assert provider.calls == 2
    assert stats.hits == 0


@pytest.mark.asyncio
async def test_streamed_answer_is_recorded_and_replayed_through_the_openai_client() -> None:
    calls = 0
    chunks = [
        {"id": "s", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [delta]}
        for delta in (
            {"index": 0, "delta": {"role": "assistant", "content": "Отв"}, "finish_reason": None},
            {"index": 0, "delta": {"content": "ет"}, "finish_reason": "stop"},
        )
    ]
    usage = {
        "id": "s",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
    }
    stream = "".join(f"data: {json.dumps(item, ensure_ascii=False)}\n\n" for item in [*chunks, usage])
    stream += "data: [DONE]\n\n"

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream.encode())

    store = MemoryStore()
    llm = ChatOpenAI(
        model_name="m",
        openai_api_key=SecretStr("test"),
        openai_api_base="http://llm.test/v1",
        max_retries=0,
        stream_usage=True,
        http_async_client=wrap_async_client(httpx.AsyncClient(transport=httpx.MockTransport(provider))),
    )

    async def answer() -> str:
        return "".join([str(chunk.content) async for chunk in llm.astream("вопрос")])

    async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
        assert await answer() == "Ответ"
        assert await answer() == "Ответ"

    assert calls == 1
    assert stats.hits == 1
    assert (stats.saved_input_tokens, stats.saved_output_tokens) == (20, 2)


def test_llm_clients_keep_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
    async_llm = llm_async_http_client()

    async_mounts = [transport for transport in async_llm._mounts.values() if transport is not None]
    assert async_mounts
    assert all(isinstance(transport, ReplayAsyncTransport) for transport in async_mounts)
    assert isinstance(async_llm._transport, ReplayAsyncTransport)


def test_usage_is_read_from_plain_and_streamed_bodies() -> None:
    assert response_usage(json.dumps(COMPLETION).encode()) == (11, 3)
    streamed = b'data: {"choices": []}\n\ndata: {"usage": {"prompt_tokens": 5, "completion_tokens": 1}}\n\n'
    assert response_usage(streamed + b"data: [DONE]\n\n") == (5, 1)
    assert response_usage(b"not json") == (None, None)


def stats_scope(seen: list[LLMReplayMode]) -> Any:
    @asynccontextmanager
    async def scope(mode: LLMReplayMode) -> AsyncIterator[LLMReplayStats]:
        seen.append(mode)
        yield LLMReplayStats(mode=mode, hits=4, misses=1, recorded=1)

    return scope


@pytest.mark.asyncio
async def test_sweep_full_evaluation_runs_inside_the_sweep_replay_mode() -> None:
    seen: list[LLMReplayMode] = []
    benchmark = SimpleNamespace(run=AsyncMock(side_effect=lambda **_: {"status": "done"}))
    engine = build_sweep_engine(
        FakeUnitOfWorkFactory(), benchmark_service=benchmark, replay=stats_scope(seen)
    )

    summary = await engine.run_full_benchmark([], "judge", run=SweepRun(replay_mode=LLMReplayMode.RECORD))
    plain = await engine.run_full_benchmark([], "judge", run=SweepRun())

    assert seen == [LLMReplayMode.RECORD, LLMReplayMode.OFF]
    assert summary["llm_replay"]["hits"] == 4
    assert summary["llm_replay"]["mode"] == "record"
    assert "llm_replay" not in plain


@pytest.mark.asyncio
async def test_sweep_keeps_the_requested_replay_mode() -> None:
    factory = FakeUnitOfWorkFactory()
    request = SweepCreateRequest(
        search_space={"top_k": {"values": [4]}}, llm_replay_mode=LLMReplayMode.REPLAY_ONLY
    )
    saved = await BenchmarkSweepService(factory).create(sweep_create_to_dto(request))

    assert saved.llm_replay_mode == LLMReplayMode.REPLAY_ONLY
    assert sweep_to_response(saved).llm_replay_mode == LLMReplayMode.REPLAY_ONLY
    assert SweepCreateDTO().llm_replay_mode == LLMReplayMode.OFF.value


@pytest.mark.asyncio
async def test_benchmark_job_records_replay_counters_in_the_run() -> None:
    seen: list[LLMReplayMode] = []
    results = SimpleNamespace(
        get_job_result=AsyncMock(return_value=None),
        record_result=AsyncMock(side_effect=lambda run: run),
    )
    benchmark = SimpleNamespace(
        run=AsyncMock(return_value={"results": [], "status": "done", "total_time_sec": 1.0, "avg_mrr": 0.5})
    )

    @asynccontextmanager
    async def corpus() -> AsyncIterator[dict[str, Any]]:
        yield {}

    service = BenchmarkJobService(
        FakeUnitOfWorkFactory(),
        benchmark,  # type: ignore[arg-type]
        results,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        corpus,
        replay=stats_scope(seen),
    )
    service_questions = AsyncMock(return_value=[])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("application.benchmark.jobs.load_benchmark_questions", service_questions)
        await service.run(
            dataset="main", top_k=4, judge_model="judge", job_id=1, llm_replay_mode=LLMReplayMode.RECORD
        )

    run = results.record_result.call_args.args[0]
    assert seen == [LLMReplayMode.RECORD]
    assert run.summary_metrics["llm_replay"]["hits"] == 4
    assert run.summary_metrics["avg_mrr"] == 0.5


def test_queued_benchmark_payload_defaults_to_no_replay() -> None:
    payload = {"dataset": "main", "top_k": 4, "judge_model": "judge"}
    assert BenchmarkPayload.model_validate(payload).llm_replay_mode == LLMReplayMode.OFF
    replayed = BenchmarkPayload.model_validate({**payload, "llm_replay_mode": "record"})
    assert replayed.llm_replay_mode == LLMReplayMode.RECORD


PRIVATE = "фрагмент приватного документа"


@pytest.mark.asyncio
async def test_store_failure_is_logged_without_the_recorded_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = Provider()
    store = MemoryStore()
    failure = DBAPIError("INSERT INTO llm_replay_entries", {"body": PRIVATE}, Exception(PRIVATE))
    store.load = AsyncMock(side_effect=failure)  # type: ignore[method-assign]
    store.save = AsyncMock(side_effect=failure)  # type: ignore[method-assign]
    client = async_client(provider)

    with caplog.at_level(logging.WARNING):
        async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
            await client.post(URL, json=body())

    assert stats.store_errors == 2
    assert caplog.records
    assert PRIVATE not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
async def test_postgres_store_error_does_not_carry_statement_parameters() -> None:
    store = PostgresReplayStore(SimpleNamespace())  # type: ignore[arg-type]
    store.upsert = AsyncMock(  # type: ignore[method-assign]
        side_effect=DBAPIError("INSERT INTO llm_replay_entries", {"body": PRIVATE}, Exception(PRIVATE))
    )

    with pytest.raises(ReplayStoreError) as raised:
        await store.save("k", "m", RecordedResponse(200, {}, PRIVATE.encode()))

    assert PRIVATE not in "".join(traceback.format_exception(raised.value))


@pytest.mark.asyncio
async def test_epoch_change_stops_reusing_earlier_recordings() -> None:
    provider, store = Provider(), MemoryStore()
    client = async_client(provider)

    async with llm_replay_scope(store, LLMReplayMode.RECORD, epoch="1"):
        await client.post(URL, json=body())

    async with llm_replay_scope(store, LLMReplayMode.RECORD, epoch="2") as stats:
        await client.post(URL, json=body())

    assert provider.calls == 2
    assert stats.hits == 0


def test_replay_marker_is_added_to_plain_and_streamed_usage_only() -> None:
    marked = json.loads(mark_replayed(json.dumps(COMPLETION).encode()))
    assert marked["usage"]["replayed"] is True
    streamed = b'data: {"choices": []}\n\ndata: {"usage": {"prompt_tokens": 5}}\n\ndata: [DONE]\n\n'
    lines = mark_replayed(streamed).decode().split("\n")
    assert json.loads(lines[0].removeprefix("data:")) == {"choices": []}
    assert json.loads(lines[2].removeprefix("data:"))["usage"]["replayed"] is True
    assert lines[4] == "data: [DONE]"
    assert mark_replayed(b'{"no": "usage"}') == b'{"no": "usage"}'


@pytest.mark.asyncio
async def test_replayed_call_is_neither_billed_nor_counted_as_paid_tokens() -> None:
    paid = {**COMPLETION, "usage": {**USAGE, "cost": 0.01}}
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=paid))
    client = AsyncOpenAI(
        api_key="test",
        base_url="http://llm.test/v1",
        http_client=wrap_async_client(httpx.AsyncClient(transport=transport)),
    )
    records: list[dict[str, Any]] = []
    token = active_llm_usage.set(records)
    try:
        async with llm_replay_scope(MemoryStore(), LLMReplayMode.RECORD):
            for _ in range(2):
                response = await client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "вопрос"}]
                )
                record_model_response(response, operation="judge")
    finally:
        active_llm_usage.reset(token)

    summary = summarize_usage(records)
    assert records[1]["replayed"] is True
    assert (records[1]["replayed_input_tokens"], records[1]["replayed_output_tokens"]) == (11, 3)
    assert summary["cost_usd"] == 0.01
    assert (summary["input_tokens"], summary["output_tokens"]) == (11, 3)
    assert summary["replayed_calls"] == 1


@pytest.mark.asyncio
async def test_replayed_stream_is_flagged_through_raw_usage_capture() -> None:
    stream = (
        'data: {"id":"s","object":"chat.completion.chunk","created":0,"model":"m",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":"Ответ"},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"s","object":"chat.completion.chunk","created":0,"model":"m","choices":[],'
        '"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22,"cost":0.02}}\n\n'
        "data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream)
    )
    llm = ChatOpenAI(
        model_name="m",
        openai_api_key=SecretStr("test"),
        openai_api_base="http://llm.test/v1",
        max_retries=0,
        stream_usage=True,
        http_async_client=wrap_async_client(httpx.AsyncClient(transport=transport)),
    )
    llm.async_client = UsageCapturingCompletions(llm.async_client)
    records: list[dict[str, Any]] = []
    token = active_llm_usage.set(records)
    try:
        async with llm_replay_scope(MemoryStore(), LLMReplayMode.RECORD):
            for _ in range(2):
                raw: dict[str, Any] = {}
                capture = active_stream_response.set(raw)
                try:
                    _ = [chunk async for chunk in llm.astream("вопрос")]
                finally:
                    active_stream_response.reset(capture)
                record_model_response(SimpleNamespace(**raw), operation="generate")
    finally:
        active_llm_usage.reset(token)

    assert [record.get("replayed", False) for record in records] == [False, True]
    assert [record["cost_usd"] for record in records] == [0.02, 0.0]
    assert records[1]["input_tokens"] == 0


def case(case_id: int, latency: float, *, replayed: bool) -> dict[str, Any]:
    return {
        "id": case_id,
        "question": "q",
        "answer": "a",
        "latency_sec": latency,
        "rag_latency_sec": latency / 2,
        "llm_replayed": replayed,
        "retriever_metrics": {"hit_rate": 1, "mrr": 1, "avg_similarity": 0.5},
        "generator_metrics": {"faithfulness": 8, "relevancy": 8, "correctness": 8},
        "cost_usd": 0.0,
    }


def test_replayed_cases_are_excluded_from_latency_percentiles_and_flagged() -> None:
    summary = compute_summary_from_results([case(1, 10.0, replayed=False), case(2, 0.1, replayed=True)])

    assert summary["latency_p50"] == summary["latency_min"] == 10.0
    assert summary["rag_latency_p50"] == 5.0
    assert summary["latency_replayed_excluded"] == 1
    assert summary["total_time_sec"] == 10.1
    assert [row["llm_replayed"] for row in summary["results"]] == [False, True]
    assert encode_csv(None).decode().split(",").index("llm_replayed") > 0
    assert ",True," in encode_csv(summary["results"][1]).decode()
    replayed_only = compute_summary_from_results([case(1, 0.1, replayed=True)])
    assert replayed_only["latency_p50"] is None
    assert replayed_only["rag_latency_p50"] is None
