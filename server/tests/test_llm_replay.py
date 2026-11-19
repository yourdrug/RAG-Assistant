"""Benchmark LLM replay: only identical requests are reused, and only inside a benchmark scope."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import pytest

from application.benchmark.dto import SweepCreateDTO
from application.benchmark.jobs import BenchmarkJobService
from application.benchmark.services import BenchmarkSweepService
from application.benchmark.sweep import SweepRun
from composition.service_providers import build_sweep_engine
from domain.benchmark.llm_replay import LLMReplayMode, LLMReplayStats
from infrastructure.ml.replay.recording import RecordedResponse, request_key, response_usage
from infrastructure.ml.replay.session import llm_replay_scope
from infrastructure.ml.replay.transport import (
    LLMReplayMissError,
    ReplayAsyncTransport,
    ReplaySyncTransport,
    llm_async_http_client,
    llm_sync_http_client,
    wrap_async_client,
    wrap_sync_client,
)
from infrastructure.worker.registry import BenchmarkPayload
from presentation.api.helpers import sweep_create_to_dto, sweep_to_response
from presentation.api.schemas.benchmark import SweepCreateRequest
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


URL = "http://llm.test/v1/chat/completions"
COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ответ"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
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
    assert first.json() == second.json() == COMPLETION
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


@pytest.mark.asyncio
async def test_sync_client_in_a_worker_thread_uses_the_benchmark_scope() -> None:
    provider, store = Provider(), MemoryStore()
    client = wrap_sync_client(httpx.Client(transport=httpx.MockTransport(provider)))

    async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
        for _ in range(2):
            response = await asyncio.to_thread(client.post, URL, json=body())
            assert response.json() == COMPLETION

    assert provider.calls == 1
    assert stats.hits == 1
    assert stats.recorded == 1


@pytest.mark.asyncio
async def test_sync_client_on_the_event_loop_thread_is_not_replayed() -> None:
    provider, store = Provider(), MemoryStore()
    client = wrap_sync_client(httpx.Client(transport=httpx.MockTransport(provider)))

    async with llm_replay_scope(store, LLMReplayMode.RECORD) as stats:
        client.post(URL, json=body())

    assert provider.calls == 1
    assert stats.recorded == 0


def test_llm_clients_keep_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
    async_llm = llm_async_http_client()
    sync_llm = llm_sync_http_client()

    async_mounts = [transport for transport in async_llm._mounts.values() if transport is not None]
    sync_mounts = [transport for transport in sync_llm._mounts.values() if transport is not None]
    assert async_mounts
    assert sync_mounts
    assert all(isinstance(transport, ReplayAsyncTransport) for transport in async_mounts)
    assert all(isinstance(transport, ReplaySyncTransport) for transport in sync_mounts)
    assert isinstance(async_llm._transport, ReplayAsyncTransport)
    assert isinstance(sync_llm._transport, ReplaySyncTransport)


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
