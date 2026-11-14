"""Local (OpenAI-compatible) LLM: judge routing, auxiliary model and server capacity checks."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import SecretStr
import pytest

from application.runtime.enums import LLMProvider
from config import LLMSection, settings
from infrastructure.benchmark.judging import clients as judge_clients
from infrastructure.ml.config.client_switch import local_llm_capacity_errors


LOCAL = {
    "llm_provider": LLMProvider.OPENAI_COMPATIBLE.value,
    "openai_compatible_base_url": "http://vllm:8000/v1",
    "openai_compatible_model": "big-local",
    "openai_compatible_api_key": SecretStr(""),
    "openrouter_api_key": SecretStr("openrouter-secret"),
}


@pytest.fixture
def local_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in LOCAL.items():
        monkeypatch.setattr(settings, key, value)

    monkeypatch.setattr(settings, "openai_compatible_auxiliary_model", "")
    monkeypatch.setattr(settings, "benchmark_judge_model", "")
    monkeypatch.setattr(judge_clients, "_judge_client_cache", {})


def capture_openai(monkeypatch: pytest.MonkeyPatch, name: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    original = getattr(judge_clients, name)

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(judge_clients, name, create)
    return calls


@pytest.mark.usefixtures("local_llm")
def test_sync_judge_uses_local_endpoint_without_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_openai(monkeypatch, "OpenAI")
    paid = []
    monkeypatch.setattr(judge_clients, "PaidSyncTransport", lambda **kwargs: paid.append(kwargs))

    client = judge_clients.get_judge_client(judge_clients.get_judge_model())

    try:
        assert calls[0]["base_url"] == "http://vllm:8000/v1"
        # Question, answer and context must never leave for OpenRouter on a local setup.
        assert calls[0]["api_key"] == "EMPTY"
        assert paid == []
    finally:
        client.retire()


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_llm")
async def test_async_judge_uses_local_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_openai(monkeypatch, "AsyncOpenAI")
    monkeypatch.setattr(settings, "openai_compatible_api_key", SecretStr("local-key"))

    async with judge_clients.create_async_judge_client():
        pass

    assert calls[0]["base_url"] == "http://vllm:8000/v1"
    assert calls[0]["api_key"] == "local-key"
    assert isinstance(calls[0]["http_client"], httpx.AsyncClient)


def test_openrouter_judge_stays_metered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", LLMProvider.OPENROUTER.value)
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr("openrouter-secret"))
    base_url, api_key, metered = judge_clients.judge_endpoint(settings)
    assert (base_url, api_key, metered) == (settings.openrouter_base_url, "openrouter-secret", True)


@pytest.mark.parametrize("auxiliary, expected", [("", "big-local"), ("small-local", "small-local")])
@pytest.mark.usefixtures("local_llm")
def test_local_auxiliary_model_drives_gates_and_judge(
    monkeypatch: pytest.MonkeyPatch, auxiliary: str, expected: str
) -> None:
    monkeypatch.setattr(settings, "openai_compatible_auxiliary_model", auxiliary)
    assert settings.active_auxiliary_model == expected
    assert judge_clients.get_judge_model() == expected


def test_openrouter_auxiliary_gates_keep_main_model() -> None:
    llm = LLMSection(openrouter_model="main", openai_compatible_auxiliary_model="ignored")
    assert llm.active_auxiliary_model == "main"


def local_candidate(**overrides: Any) -> Any:
    return settings.model_copy(
        update={
            **LOCAL,
            "openai_compatible_auxiliary_model": "",
            "llm_num_ctx_narrow": 8192,
            "llm_num_ctx_broad": 16384,
            **overrides,
        }
    )


def test_capacity_rejects_window_smaller_than_broad_budget() -> None:
    errors = local_llm_capacity_errors(
        local_candidate(), {"data": [{"id": "big-local", "max_model_len": 8192}]}
    )
    assert len(errors) == 1
    assert "16384" in errors[0]


def test_capacity_accepts_matching_window_and_unknown_window() -> None:
    assert not local_llm_capacity_errors(
        local_candidate(), {"data": [{"id": "big-local", "max_model_len": 16384}]}
    )
    # Ollama / LM Studio do not report a window: only the model id is checked.
    assert not local_llm_capacity_errors(local_candidate(), {"data": [{"id": "big-local"}]})


def test_capacity_requires_auxiliary_model_to_be_served() -> None:
    errors = local_llm_capacity_errors(
        local_candidate(openai_compatible_auxiliary_model="small-local"),
        {"data": [{"id": "big-local", "max_model_len": 32768}]},
    )
    assert errors == ["OpenAI-compatible server does not serve model 'small-local'"]


def test_capacity_ignores_unexpected_payload() -> None:
    assert local_llm_capacity_errors(local_candidate(), {"object": "list"}) == []
    assert local_llm_capacity_errors(local_candidate(), []) == []
