"""OpenAI-compatible LLM endpoint: judge routing, auxiliary model and server capacity checks."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import SecretStr
import pytest

from config import LLMSection, settings
from domain.shared.exceptions import ValidationError
from infrastructure.benchmark.judging import clients as judge_clients
from infrastructure.ml.clients.factories import normalize_llm_model
from infrastructure.ml.config.client_switch import llm_capacity_errors, probe_llm_endpoint


LOCAL = {
    "llm_base_url": "http://vllm:8000/v1",
    "llm_model": "big-local",
    "llm_api_key": SecretStr(""),
}


@pytest.fixture
def local_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in LOCAL.items():
        monkeypatch.setattr(settings, key, value)

    monkeypatch.setattr(settings, "llm_auxiliary_model", "")
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
def test_sync_judge_uses_configured_endpoint_with_placeholder_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_openai(monkeypatch, "OpenAI")

    client = judge_clients.get_judge_client(judge_clients.get_judge_model())

    try:
        assert calls[0]["base_url"] == "http://vllm:8000/v1"
        # Question, answer and context must never leave the configured endpoint.
        assert calls[0]["api_key"] == "EMPTY"
    finally:
        client.retire()


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_llm")
async def test_async_judge_uses_local_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_openai(monkeypatch, "AsyncOpenAI")
    monkeypatch.setattr(settings, "llm_api_key", SecretStr("local-key"))

    async with judge_clients.create_async_judge_client():
        pass

    assert calls[0]["base_url"] == "http://vllm:8000/v1"
    assert calls[0]["api_key"] == "local-key"
    assert isinstance(calls[0]["http_client"], httpx.AsyncClient)


def test_cloud_judge_uses_cloud_endpoint_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "llm_api_key", SecretStr("cloud-secret"))
    assert judge_clients.judge_endpoint(settings) == ("https://openrouter.ai/api/v1", "cloud-secret")


@pytest.mark.parametrize("auxiliary, expected", [("", "big-local"), ("small-local", "small-local")])
@pytest.mark.usefixtures("local_llm")
def test_local_auxiliary_model_drives_gates_and_judge(
    monkeypatch: pytest.MonkeyPatch, auxiliary: str, expected: str
) -> None:
    monkeypatch.setattr(settings, "llm_auxiliary_model", auxiliary)
    assert settings.active_auxiliary_model == expected
    assert judge_clients.get_judge_model() == expected


def test_empty_auxiliary_model_reuses_main_model() -> None:
    assert LLMSection(llm_model="main").active_auxiliary_model == "main"
    assert LLMSection(llm_model="main", llm_auxiliary_model="small").active_auxiliary_model == "small"


def local_candidate(**overrides: Any) -> Any:
    return settings.model_copy(
        update={
            **LOCAL,
            "llm_auxiliary_model": "",
            "llm_num_ctx_narrow": 8192,
            "llm_num_ctx_broad": 16384,
            **overrides,
        }
    )


def test_capacity_rejects_window_smaller_than_broad_budget() -> None:
    errors = llm_capacity_errors(local_candidate(), {"data": [{"id": "big-local", "max_model_len": 8192}]})
    assert len(errors) == 1
    assert "16384" in errors[0]


def test_capacity_accepts_matching_window_and_unknown_window() -> None:
    assert not llm_capacity_errors(local_candidate(), {"data": [{"id": "big-local", "max_model_len": 16384}]})
    # Ollama / LM Studio do not report a window: only the model id is checked.
    assert not llm_capacity_errors(local_candidate(), {"data": [{"id": "big-local"}]})


def test_capacity_requires_auxiliary_model_to_be_served() -> None:
    errors = llm_capacity_errors(
        local_candidate(llm_auxiliary_model="small-local"),
        {"data": [{"id": "big-local", "max_model_len": 32768}]},
    )
    assert errors == ["LLM endpoint does not serve model 'small-local'"]


def test_capacity_ignores_unexpected_payload() -> None:
    assert llm_capacity_errors(local_candidate(), {"object": "list"}) == []
    assert llm_capacity_errors(local_candidate(), []) == []


def test_models_listing_normalizes_openrouter_and_vllm_entries() -> None:
    assert normalize_llm_model(
        {
            "id": "a/b",
            "name": "A B",
            "context_length": 8192,
            "pricing": {"prompt": "0.000001", "completion": "2e-6"},
        }
    ) == {"id": "a/b", "name": "A B", "context_length": 8192, "pricing": {"prompt": 1.0, "completion": 2.0}}
    assert normalize_llm_model({"id": "Qwen/Qwen2.5-3B", "max_model_len": 16384}) == {
        "id": "Qwen/Qwen2.5-3B",
        "name": "Qwen/Qwen2.5-3B",
        "context_length": 16384,
        "pricing": None,
    }
    assert normalize_llm_model({"id": "text-embedding-3-small"}) is None
    assert normalize_llm_model({"name": "no id"}) is None


def probe_with(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> Any:
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real(**kwargs, transport=httpx.MockTransport(lambda _request: response)),
    )
    candidate = local_candidate(llm_model="manual-model")
    return probe_llm_endpoint(candidate, {"llm_model"}, {"cached_llm": object()})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [httpx.Response(404), httpx.Response(405), httpx.Response(501), httpx.Response(200, text="not json")],
)
async def test_endpoint_without_models_listing_accepts_manual_model(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response
) -> None:
    await probe_with(monkeypatch, response)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 503])
async def test_endpoint_auth_and_server_errors_still_block_changes(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        await probe_with(monkeypatch, httpx.Response(status))


@pytest.mark.asyncio
async def test_listed_models_are_still_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="manual-model"):
        await probe_with(monkeypatch, httpx.Response(200, json={"data": [{"id": "other"}]}))
