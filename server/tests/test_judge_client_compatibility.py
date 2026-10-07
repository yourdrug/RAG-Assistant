"""Exercise real Instructor provider validation and request/response handling offline."""

import json

import httpx
import instructor
import pytest
from openai import OpenAI

from domain.value_objects.llm_provider import LLMProvider
from infrastructure.benchmark import judge
from infrastructure.benchmark.token_usage import judge_usage


@pytest.mark.parametrize(
    ("provider", "base_url", "mode"),
    [
        (LLMProvider.OPENROUTER, "https://openrouter.ai/api/v1", instructor.Mode.JSON),
        (LLMProvider.OLLAMA, "http://localhost:11434/v1", instructor.Mode.MD_JSON),
    ],
)
def test_judge_client_provider_compatibility(monkeypatch, provider, base_url, mode):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "judge-test",
                "object": "chat.completion",
                "created": 0,
                "model": "judge-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"score": 8, "reason": "grounded"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

    monkeypatch.setattr(judge.settings, "llm_provider", provider)
    monkeypatch.setattr(judge.settings, "openrouter_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(judge.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(judge.settings, "ollama_base_url", "http://localhost:11434")
    monkeypatch.setattr(judge, "_judge_client_cache", {})
    records = []
    token = judge_usage.set(records)
    with OpenAI(
        base_url=base_url,
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    ) as raw:

        def create_raw(**kwargs):
            assert kwargs["base_url"] == base_url
            assert kwargs["max_retries"] == 0
            return raw

        monkeypatch.setattr(judge, "OpenAI", create_raw)
        try:
            client = judge.get_judge_client("judge-model")
            assert client.mode == mode
            assert judge.get_judge_client("judge-model") is client
            score = judge.judge_with_structured_output(client, "Evaluate the answer", "judge-model")
        finally:
            judge_usage.reset(token)

    assert score.score == 8
    assert score.reason == "grounded"
    assert len(requests) == 1
    assert "tools" not in requests[0]
    assert requests[0]["model"] == "judge-model"
    if provider == LLMProvider.OPENROUTER:
        assert requests[0]["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in requests[0]
    assert records == [{"input_tokens": 20, "output_tokens": 10}]
