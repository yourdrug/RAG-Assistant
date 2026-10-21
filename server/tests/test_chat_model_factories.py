"""Role-specific model options survive consolidation of chat factories."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from config import Settings, settings
from domain.runtime.enums import LLMProvider
from infrastructure.benchmark import judge, retrieval
from infrastructure.ml.clients import factories


@pytest.fixture
def constructors(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    ollama, openrouter = Mock(), Mock()
    monkeypatch.setattr(factories, "ChatOllama", ollama)
    monkeypatch.setattr(factories, "ChatOpenAI", openrouter)
    http_client = Mock(return_value="sync")
    async_http_client = Mock(return_value="async")
    monkeypatch.setattr(httpx, "Client", http_client)
    monkeypatch.setattr(httpx, "AsyncClient", async_http_client)
    monkeypatch.setattr(factories, "PaidSyncTransport", Mock(return_value="paid-sync"))
    monkeypatch.setattr(factories, "PaidAsyncTransport", Mock(return_value="paid-async"))
    return SimpleNamespace(
        ollama=ollama, openrouter=openrouter, http_client=http_client, async_http_client=async_http_client
    )


@pytest.fixture
def model_settings() -> Settings:
    return settings.model_copy(
        update={
            "llm_model": "local",
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
    constructors.ollama.assert_not_called()
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


@pytest.mark.parametrize("role", list(factories.ChatModelRole))
def test_ollama_roles_preserve_options(
    constructors: SimpleNamespace, model_settings: Settings, role: factories.ChatModelRole
) -> None:
    factories.build_chat_model(LLMProvider.OLLAMA, role, model=model_settings)
    is_broad = role == factories.ChatModelRole.BROAD
    benchmark = role == factories.ChatModelRole.BENCHMARK
    constructors.openrouter.assert_not_called()
    options = constructors.ollama.call_args.kwargs
    assert options["model"] == "local"
    assert options["temperature"] == (0 if benchmark else 0.3)

    if benchmark:
        assert set(options) == {"model", "base_url", "temperature"}
    else:
        assert options["num_ctx"] == (444 if is_broad else 333)
        assert options["top_p"] == 0.8
        assert options["client_kwargs"] == {"timeout": 55}

    if role in (factories.ChatModelRole.NARROW, factories.ChatModelRole.BROAD):
        assert options["num_predict"] == (222 if is_broad else 111)
    else:
        assert "num_predict" not in options


@pytest.mark.parametrize("provider", list(LLMProvider))
def test_factory_overrides_and_breadth_fallback(constructors: SimpleNamespace, provider: LLMProvider) -> None:
    model = settings.model_copy(update={"llm_provider": provider})
    factories.create_llm(temperature=0, top_p=0, num_ctx=0, num_predict=0, model=model)
    options = (
        constructors.openrouter if provider == LLMProvider.OPENROUTER else constructors.ollama
    ).call_args.kwargs
    assert options["temperature"] == 0

    if provider == LLMProvider.OPENROUTER:
        assert options["max_tokens"] == 0
    else:
        assert options["num_ctx"] == 0
        assert options["top_p"] == 0

    factories.create_llm_for_breadth(
        "broad",
        num_ctx_broad=0,
        num_predict_broad=0,
        model=model,
    )
    options = (
        constructors.openrouter if provider == LLMProvider.OPENROUTER else constructors.ollama
    ).call_args.kwargs

    if provider == LLMProvider.OPENROUTER:
        assert options["max_tokens"] == model.llm_num_predict_broad
    else:
        assert options["num_ctx"] == model.llm_num_ctx_broad
        assert options["num_predict"] == model.llm_num_predict_broad


@pytest.mark.parametrize("provider", list(LLMProvider))
def test_benchmark_preserves_explicit_model_and_local_endpoint(
    constructors: SimpleNamespace, provider: LLMProvider
) -> None:
    retrieval.build_llm("benchmark-model", "http://benchmark:11434", provider)

    if provider == LLMProvider.OPENROUTER:
        assert constructors.openrouter.call_args.kwargs["model_name"] == "benchmark-model"
    else:
        assert constructors.ollama.call_args.kwargs == {
            "model": "benchmark-model",
            "base_url": "http://benchmark:11434",
            "temperature": 0,
        }


def test_judge_and_auxiliary_share_default_model(
    monkeypatch: pytest.MonkeyPatch, constructors: SimpleNamespace
) -> None:
    monkeypatch.setattr(settings, "llm_provider", LLMProvider.OPENROUTER)
    monkeypatch.setattr(settings, "benchmark_judge_model", "")
    factories.create_fast_llm_for_auxiliary()
    assert judge.get_judge_model() == constructors.openrouter.call_args.kwargs["model_name"]
