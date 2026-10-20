"""Characterize the public judge factory before changing its cached pool ownership."""

import httpx
import asyncio
import pytest
from openai import OpenAI

from config import settings
from infrastructure.benchmark import judge


@pytest.fixture
def judge_http(monkeypatch):
    requests = []

    def respond(request):
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

    def create(**kwargs):
        pool = httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=False)
        pools.append(pool)
        kwargs["http_client"] = pool
        return OpenAI(**kwargs)

    monkeypatch.setattr(judge, "OpenAI", create)
    monkeypatch.setattr(judge, "_judge_client_cache", {})
    yield requests, pools
    for pool in pools:
        pool.close()


def test_judge_client_reuses_model_and_returns_validated_score(judge_http, monkeypatch):
    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama:11434")
    client = judge.get_judge_client("test-model")
    assert judge.get_judge_client("test-model") is client
    result = judge.judge_with_structured_output(client, "evaluate", "test-model")
    assert result.score == 7
    assert result.reason == "supported"
    assert judge_http[0][0].url.host == "ollama"


def test_next_judge_call_uses_new_endpoint_and_closes_idle_previous_pool(judge_http, monkeypatch):
    monkeypatch.setattr(settings, "ollama_base_url", "http://old:11434")
    old = judge.get_judge_client("test-model")
    assert judge.judge_with_structured_output(old, "evaluate", "test-model").score == 7
    monkeypatch.setattr(settings, "ollama_base_url", "http://new:11434")
    new = judge.get_judge_client("test-model")
    assert judge.judge_with_structured_output(new, "evaluate", "test-model").score == 7
    assert judge_http[0][-1].url.host == "new"
    assert judge_http[1][0].is_closed


@pytest.mark.asyncio
async def test_queued_judge_preserves_client_until_worker_admission(judge_http, monkeypatch):
    queued, release = asyncio.Event(), asyncio.Event()
    real_thread = asyncio.to_thread

    async def thread(function, *args, **kwargs):
        if function is judge.judge_with_structured_output and not queued.is_set():
            queued.set()
            await release.wait()
        return await real_thread(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", thread)
    monkeypatch.setattr(settings, "ollama_base_url", "http://old:11434")
    call = asyncio.create_task(judge.judge_answer_async("question", "answer", "context"))
    await queued.wait()
    try:
        monkeypatch.setattr(settings, "ollama_base_url", "http://new:11434")
        judge.retire_judge_clients()

        release.set()
        result = await call
        assert result["faithfulness"] == 7
        assert judge_http[0][0].url.host == "old"
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        await judge.close_judge_clients()


@pytest.mark.asyncio
async def test_container_shutdown_waits_for_owned_benchmark_pool_cleanup(judge_http):
    from composition.container import Container

    judge.get_judge_client("test-model")
    assert not judge_http[1][0].is_closed
    await Container().infrastructure.dispose()
    assert judge_http[1][0].is_closed


def test_judge_constructor_cannot_mix_old_origin_with_new_provider_credentials(judge_http, monkeypatch):
    from domain.runtime.enums import LLMProvider

    class UpdatingLock:
        def __enter__(self):
            monkeypatch.setattr(settings, "llm_provider", LLMProvider.OPENROUTER.value)
            monkeypatch.setattr(settings, "openrouter_api_key", "private-test-credential")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(settings, "llm_provider", LLMProvider.OLLAMA.value)
    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama:11434")
    monkeypatch.setattr(judge, "_judge_client_lock", UpdatingLock())
    client = judge.get_judge_client("test-model")
    result = judge.judge_with_structured_output(client, "evaluate", "test-model")
    assert result.score == 7
    request = judge_http[0][0]
    assert request.url.host == "ollama"
    assert request.headers["Authorization"] == f"Bearer {LLMProvider.OLLAMA.value}"
