"""Role-specific model options survive consolidation of chat factories."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from application.runtime.enums import LLMProvider
from config import Settings, settings
from infrastructure.benchmark.judging import clients as judge_clients
from infrastructure.ml.clients import factories


@pytest.fixture
def constructors(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    openrouter = Mock()
    monkeypatch.setattr(factories, "ChatOpenAI", openrouter)
    http_client = Mock(return_value="sync")
    async_http_client = Mock(return_value="async")
    monkeypatch.setattr(httpx, "Client", http_client)
    monkeypatch.setattr(httpx, "AsyncClient", async_http_client)
    monkeypatch.setattr(factories, "PaidSyncTransport", Mock(return_value="paid-sync"))
    monkeypatch.setattr(factories, "PaidAsyncTransport", Mock(return_value="paid-async"))
    return SimpleNamespace(
        openrouter=openrouter, http_client=http_client, async_http_client=async_http_client
    )


@pytest.fixture
def model_settings() -> Settings:
    return settings.model_copy(
        update={
            "openrouter_model": "remote",
            "openrouter_api_key": "test",
            "llm_temperature": 0.3,
            "llm_top_p": 0.8,
            "llm_num_predict_narrow": 111,
            "llm_num_predict_broad": 222,
            "llm_num_ctx_narrow": 333,
            "llm_num_ctx_broad": 444,
            "llm_request_timeout": 55,
        }
    )


@pytest.mark.parametrize("role", list(factories.ChatModelRole))
def test_openrouter_roles_preserve_options(
    constructors: SimpleNamespace, model_settings: Settings, role: factories.ChatModelRole
) -> None:
    factories.build_chat_model(LLMProvider.OPENROUTER, role, model=model_settings)
    is_broad = role == factories.ChatModelRole.BROAD
    auxiliary = role == factories.ChatModelRole.AUXILIARY
    benchmark = role == factories.ChatModelRole.BENCHMARK
    options = constructors.openrouter.call_args.kwargs
    assert options["model_name"] == (factories.FAST_OPENROUTER_MODEL if auxiliary else "remote")
    assert options["max_tokens"] == (200 if auxiliary else 222 if is_broad or benchmark else 111)
    assert options["max_retries"] == 0
    assert options["http_client"] == "sync"
    assert options["http_async_client"] == "async"
    assert options["temperature"] == (0 if auxiliary or benchmark else 0.3)

    if not benchmark:
        assert options["request_timeout"] == (10 if auxiliary else 120)
    else:
        assert "request_timeout" not in options

    assert options.get("stream_usage", False) == (not auxiliary and not benchmark)
    constructors.http_client.assert_called_once_with(follow_redirects=False, transport="paid-sync")
    constructors.async_http_client.assert_called_once_with(follow_redirects=False, transport="paid-async")


def test_retired_provider_is_rejected(constructors: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        factories.build_chat_model("ollama", factories.ChatModelRole.DEFAULT)
    constructors.openrouter.assert_not_called()


@pytest.mark.parametrize("provider", list(LLMProvider))
def test_factory_overrides_and_breadth_fallback(constructors: SimpleNamespace, provider: LLMProvider) -> None:
    model = settings.model_copy(update={"llm_provider": provider})
    factories.create_llm(temperature=0, top_p=0, num_ctx=0, num_predict=0, model=model)
    options = constructors.openrouter.call_args.kwargs
    assert options["temperature"] == 0

    assert options["max_tokens"] == 0

    factories.create_llm_for_breadth(
        "broad",
        num_ctx_broad=0,
        num_predict_broad=0,
        model=model,
    )
    options = constructors.openrouter.call_args.kwargs

    assert options["max_tokens"] == model.llm_num_predict_broad


def test_judge_and_auxiliary_share_default_model(
    monkeypatch: pytest.MonkeyPatch, constructors: SimpleNamespace
) -> None:
    monkeypatch.setattr(settings, "llm_provider", LLMProvider.OPENROUTER)
    monkeypatch.setattr(settings, "benchmark_judge_model", "")
    factories.create_fast_llm_for_auxiliary()
    assert judge_clients.get_judge_model() == constructors.openrouter.call_args.kwargs["model_name"]
