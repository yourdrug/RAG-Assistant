"""Exercise real Instructor provider validation and request/response handling offline."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import instructor
from openai import OpenAI
import pytest

from application.runtime.enums import LLMProvider
from config import settings
from infrastructure.benchmark.judging import clients as judge_clients, judge
from infrastructure.benchmark.judging.token_usage import judge_usage
from infrastructure.budget.runtime import BudgetRuntime


if TYPE_CHECKING:
    from infrastructure.ml.clients.llm_schemas import JudgeScore


@pytest.mark.parametrize(
    "provider, base_url, mode",
    [
        (LLMProvider.OPENROUTER, "https://openrouter.ai/api/v1", instructor.Mode.JSON),
        (LLMProvider.OLLAMA, "http://localhost:11434/v1", instructor.Mode.MD_JSON),
    ],
)
def test_judge_client_provider_compatibility(
    monkeypatch: pytest.MonkeyPatch, provider: str, base_url: str, mode: instructor.Mode
) -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
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

    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434")
    monkeypatch.setattr(judge_clients, "_judge_client_cache", {})
    runtime = BudgetRuntime()
    records: list[dict[str, Any]] = []
    token = judge_usage.set(records)
    with OpenAI(
        base_url=base_url,
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    ) as raw:

        def create_raw(**kwargs: Any) -> OpenAI:
            assert kwargs["base_url"] == base_url
            assert kwargs["max_retries"] == 0
            return raw

        monkeypatch.setattr(judge_clients, "OpenAI", create_raw)
        try:
            client = judge_clients.get_judge_client("judge-model", runtime=runtime)
            assert client.mode == mode
            assert judge_clients.get_judge_client("judge-model", runtime=runtime) is client
            score = judge.judge_with_structured_output(client, "Evaluate the answer", "judge-model")
        finally:
            judge_usage.reset(token)

    check_judge_contract(score, requests, records, provider)


def check_judge_contract(
    score: JudgeScore, requests: list[dict[str, Any]], records: list[dict[str, Any]], provider: str
) -> None:
    assert score.score == 8
    assert score.reason == "grounded"
    assert len(requests) == 1
    assert "tools" not in requests[0]
    assert requests[0]["model"] == "judge-model"

    if provider == LLMProvider.OPENROUTER:
        assert requests[0]["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in requests[0]

    assert len(records) == 1
    assert records[0]["input_tokens"] == 20
    assert records[0]["output_tokens"] == 10
    assert records[0]["model"] == "judge-model"
    assert records[0]["cost_usd"] is None
