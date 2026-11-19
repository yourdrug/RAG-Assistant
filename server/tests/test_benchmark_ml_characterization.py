"""Characterize the public judge factory before changing its cached pool ownership."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
from openai import AsyncOpenAI
from pydantic import SecretStr
import pytest
import pytest_asyncio

from composition.container import Container
from config import settings
from infrastructure.benchmark.judging import clients as judge_clients, judge


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def eventually_closed(pool: httpx.AsyncClient) -> bool:
    """A retired pool closes in a background task once its last call has finished."""
    for _ in range(100):
        if pool.is_closed:
            return True

        await asyncio.sleep(0)

    return pool.is_closed


@pytest_asyncio.fixture
async def judge_http(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[list[Any], list[Any]]]:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"score":7,"reason":"supported"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    pools = []

    def create(**kwargs: Any) -> AsyncOpenAI:
        pool = httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=False)
        pools.append(pool)
        kwargs["http_client"] = pool
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(judge_clients, "AsyncOpenAI", create)
    monkeypatch.setattr(judge_clients, "_judge_client_cache", {})
    yield requests, pools

    for pool in pools:
        await pool.aclose()


@pytest.mark.asyncio
async def test_judge_client_reuses_model_and_returns_validated_score(
    judge_http: tuple[list[Any], list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "llm_base_url", "https://openrouter.ai/api/v1")
    client = judge_clients.get_judge_client("test-model")
    assert judge_clients.get_judge_client("test-model") is client
    result = await judge.judge_with_structured_output(client, "evaluate", "test-model")
    assert result.score == 7
    assert result.reason == "supported"
    assert judge_http[0][0].url.host == "openrouter.ai"


@pytest.mark.asyncio
async def test_next_judge_call_uses_new_endpoint_and_closes_idle_previous_pool(
    judge_http: tuple[list[Any], list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "llm_base_url", "https://old.openrouter.ai/v1")
    old = judge_clients.get_judge_client("test-model")
    assert (await judge.judge_with_structured_output(old, "evaluate", "test-model")).score == 7
    monkeypatch.setattr(settings, "llm_base_url", "https://new.openrouter.ai/v1")
    new = judge_clients.get_judge_client("test-model")
    assert (await judge.judge_with_structured_output(new, "evaluate", "test-model")).score == 7
    assert judge_http[0][-1].url.host == "new.openrouter.ai"
    assert await eventually_closed(judge_http[1][0])


@pytest.mark.asyncio
async def test_queued_judge_preserves_client_until_worker_admission(
    judge_http: tuple[list[Any], list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    queued, release = asyncio.Event(), asyncio.Event()
    real_judge = judge.judge_with_structured_output

    async def gated(client: Any, prompt: str, model: str) -> Any:
        if not queued.is_set():
            queued.set()
            await release.wait()

        return await real_judge(client, prompt, model)

    monkeypatch.setattr(judge, "judge_with_structured_output", gated)
    monkeypatch.setattr(settings, "llm_base_url", "https://old.openrouter.ai/v1")
    call = asyncio.create_task(judge.judge_answer_async("question", "answer", "context"))
    await queued.wait()
    try:
        monkeypatch.setattr(settings, "llm_base_url", "https://new.openrouter.ai/v1")
        judge_clients.retire_judge_clients()

        release.set()
        result = await call
        assert result["faithfulness"] == 7
        assert judge_http[0][0].url.host == "old.openrouter.ai"
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        await judge_clients.close_judge_clients()


@pytest.mark.asyncio
async def test_container_shutdown_closes_benchmark_judge_pool(
    judge_http: tuple[list[Any], list[Any]],
) -> None:
    container = Container()
    judge_clients.get_judge_client("test-model")
    assert len(judge_http[1]) == 1
    await container.infrastructure.dispose()
    assert judge_http[1][0].is_closed


@pytest.mark.asyncio
async def test_judge_constructor_cannot_mix_old_origin_with_new_provider_credentials(
    judge_http: tuple[list[Any], list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:

    class UpdatingLock:
        def __enter__(self) -> None:
            monkeypatch.setattr(settings, "llm_api_key", SecretStr("private-test-credential"))

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(settings, "llm_api_key", SecretStr("old-test-credential"))
    monkeypatch.setattr(settings, "llm_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(judge_clients, "_judge_client_lock", UpdatingLock())
    client = judge_clients.get_judge_client("test-model")
    result = await judge.judge_with_structured_output(client, "evaluate", "test-model")
    assert result.score == 7
    request = judge_http[0][0]
    assert request.url.host == "openrouter.ai"
    assert request.headers["Authorization"] == "Bearer old-test-credential"
