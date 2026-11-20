"""Role-specific model options survive consolidation of chat factories."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import instructor
from pydantic import SecretStr
import pytest

from config import Settings, settings
from infrastructure.benchmark.judging import clients as judge_clients
from infrastructure.ml.clients import factories, instructor_client
from infrastructure.ml.replay import transport as replay_transport


@pytest.fixture
def constructors(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    chat_openai = Mock()
    monkeypatch.setattr(factories, "ChatOpenAI", chat_openai)
    async_http_client = Mock(return_value="async")
    monkeypatch.setattr(httpx, "AsyncClient", async_http_client)
    # Replay wrapping is covered by test_llm_replay; the stubs here are not real clients.
    monkeypatch.setattr(replay_transport, "wrap_async_client", lambda client: client)
    return SimpleNamespace(chat_openai=chat_openai, async_http_client=async_http_client)


@pytest.fixture
def model_settings() -> Settings:
    return settings.model_copy(
        update={
            "llm_model": "remote",
            "llm_api_key": SecretStr("test"),
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
def test_roles_preserve_options(
    constructors: SimpleNamespace, model_settings: Settings, role: factories.ChatModelRole
) -> None:
    factories.build_chat_model(role, model=model_settings)
    is_broad = role == factories.ChatModelRole.BROAD
    auxiliary = role == factories.ChatModelRole.AUXILIARY
    benchmark = role == factories.ChatModelRole.BENCHMARK
    options = constructors.chat_openai.call_args.kwargs
    # An empty auxiliary model reuses the answer model.
    assert options["model_name"] == "remote"
    assert options["openai_api_key"].get_secret_value() == "test"
    assert options["max_tokens"] == (200 if auxiliary else 222 if is_broad or benchmark else 111)
    assert options["max_retries"] == 0
    # Sync invocation is refused by ManagedLLM, so no replay-wrapped sync client is built.
    assert "http_client" not in options
    assert options["http_async_client"] == "async"
    assert options["temperature"] == (0 if auxiliary or benchmark else 0.3)

    if not benchmark:
        assert options["request_timeout"] == (10 if auxiliary else 120)
    else:
        assert "request_timeout" not in options

    assert options.get("stream_usage", False) == (not auxiliary and not benchmark)
    assert options.get("top_p") == (None if auxiliary or benchmark else 0.8)
    constructors.async_http_client.assert_called_once_with(follow_redirects=False)


@pytest.mark.parametrize("role", list(factories.ChatModelRole))
@pytest.mark.parametrize(
    "base_url,body",
    [
        (
            "https://openrouter.ai/api/v1",
            {"reasoning": {"enabled": False}, "provider": {"ignore": ["novita"]}},
        ),
        ("http://vllm:8000/v1", {"chat_template_kwargs": {"enable_thinking": False}}),
    ],
)
def test_reasoning_is_disabled_for_answer_calls(
    constructors: SimpleNamespace,
    model_settings: Settings,
    role: factories.ChatModelRole,
    base_url: str,
    body: dict[str, object],
) -> None:
    factories.build_chat_model(role, model=model_settings.model_copy(update={"llm_base_url": base_url}))
    options = constructors.chat_openai.call_args.kwargs

    assert options.get("extra_body") == (None if role == factories.ChatModelRole.BENCHMARK else body)


def test_reasoning_switch_can_be_turned_off(constructors: SimpleNamespace, model_settings: Settings) -> None:
    model = model_settings.model_copy(update={"llm_disable_reasoning": False})
    factories.build_chat_model(factories.ChatModelRole.NARROW, model=model)

    assert "extra_body" not in constructors.chat_openai.call_args.kwargs


@pytest.mark.parametrize("role", list(factories.ChatModelRole))
@pytest.mark.parametrize("auxiliary_model", ["", "small-local"])
def test_endpoint_uses_configured_url_and_models(
    constructors: SimpleNamespace,
    model_settings: Settings,
    role: factories.ChatModelRole,
    auxiliary_model: str,
) -> None:
    model = model_settings.model_copy(
        update={
            "llm_base_url": "http://vllm:8000/v1",
            "llm_model": "Qwen/Qwen2.5-3B-Instruct",
            "llm_auxiliary_model": auxiliary_model,
        }
    )
    factories.build_chat_model(role, model=model)
    options = constructors.chat_openai.call_args.kwargs
    auxiliary = role == factories.ChatModelRole.AUXILIARY
    assert options["openai_api_base"] == "http://vllm:8000/v1"
    assert options["model_name"] == (
        (auxiliary_model or "Qwen/Qwen2.5-3B-Instruct") if auxiliary else "Qwen/Qwen2.5-3B-Instruct"
    )
    constructors.async_http_client.assert_called_once_with(follow_redirects=False)


def test_keyless_endpoint_gets_placeholder_key(
    constructors: SimpleNamespace, model_settings: Settings
) -> None:
    model = model_settings.model_copy(update={"llm_api_key": SecretStr("")})
    factories.build_chat_model(factories.ChatModelRole.DEFAULT, model=model)
    key = constructors.chat_openai.call_args.kwargs["openai_api_key"]
    assert key.get_secret_value() == instructor_client.NO_API_KEY


def test_factory_overrides_and_breadth_fallback(constructors: SimpleNamespace) -> None:
    model = settings.model_copy()
    factories.create_llm(temperature=0, top_p=0, num_ctx=0, num_predict=0, model=model)
    options = constructors.chat_openai.call_args.kwargs
    assert options["temperature"] == 0
    assert options["top_p"] == 0

    assert options["max_tokens"] == 0

    factories.create_llm_for_breadth(
        "broad",
        num_ctx_broad=0,
        num_predict_broad=0,
        model=model,
    )
    options = constructors.chat_openai.call_args.kwargs

    assert options["max_tokens"] == model.llm_num_predict_broad


def test_judge_and_auxiliary_share_default_model(
    monkeypatch: pytest.MonkeyPatch, constructors: SimpleNamespace
) -> None:
    monkeypatch.setattr(settings, "llm_auxiliary_model", "small")
    monkeypatch.setattr(settings, "benchmark_judge_model", "")
    factories.create_fast_llm_for_auxiliary()
    assert (
        judge_clients.get_judge_model() == constructors.chat_openai.call_args.kwargs["model_name"] == "small"
    )


@pytest.mark.parametrize("base_url", ["https://openrouter.ai/api/v1", "http://vllm:8000/v1"])
def test_structured_output_uses_json_mode_for_every_provider(base_url: str) -> None:
    client = instructor_client.create_async_instructor_client(base_url, "")
    assert client.mode == instructor.Mode.JSON  # type: ignore[attr-defined]  # runtime attribute
