"""ML snapshot switching at the real HTTP and resource ownership boundaries."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import SecretStr
import pytest
from qdrant_client.models import Filter

from application.admin import ConfigService
from application.runtime.config import ConfigParameter, ConfigParameterChanged, ConfigValueType
from application.runtime.dynamic_params import DYNAMIC_FIELDS_MAP
from config import Settings, settings
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.rag.breadth import Breadth
from domain.shared.exceptions import ValidationError
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.health import system_health_probe
from infrastructure.initialization import build_defaults, load_config_from_db
from infrastructure.ml.clients import factories
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.config.provider_configuration import (
    SettingsProviderConfiguration,
    validate_chunking,
    validate_provider_credentials,
)
from infrastructure.ml.rag import rag_retrieval
from infrastructure.ml.rag.rag_prompts import condense_question
from infrastructure.repositories.vector import qdrant_ops
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

    from qdrant_client.models import PointStruct

    from infrastructure.ml.clients.tei_clients import TEIEmbeddingsClient
    from tests.fakes import FakeUnitOfWork


ENDPOINT_KEYS = (
    "openrouter_base_url",
    "tei_embed_url",
    "tei_rerank_url",
    "deepinfra_base_url",
)
POOL_KEYS = (
    "embedding_http_max_connections",
    "embedding_http_max_keepalive_connections",
    "rerank_http_max_connections",
    "rerank_http_max_keepalive_connections",
)


def row(key: str, value: object) -> ConfigParameter:
    return ConfigParameter(
        key, str(value), ConfigValueType.INT if key in POOL_KEYS else ConfigValueType.STR, "ml"
    )


@pytest.fixture
def model() -> Settings:
    return Settings(
        _env_file=None,
        ml_provider="tei",
        tei_embed_url="http://tei-embed:8080",
        tei_rerank_url="http://tei-rerank:8080",
        ml_internal_allowed_origins=[
            "http://localhost:11434",
            "http://tei-embed:8080",
            "http://tei-rerank:8080",
            "http://replacement:8080",
        ],
    )


@pytest.mark.parametrize("key", ENDPOINT_KEYS + POOL_KEYS)
def test_keys_have_working_dynamic_declarations(key: str) -> None:
    assert key in DYNAMIC_FIELDS_MAP


@pytest.mark.parametrize("prefix, connections, keepalive", [("embedding", 40, 20), ("rerank", 20, 10)])
def test_factories_use_independent_effective_pool_limits(
    model: Settings, prefix: str, connections: int, keepalive: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []
    real = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        captured.append(kwargs["limits"])
        return real(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    adapter = (
        factories.create_embeddings(model) if prefix == "embedding" else factories.create_reranker(model)
    )
    assert captured[0].max_connections == connections
    assert captured[0].max_keepalive_connections == keepalive
    asyncio.run(adapter.close())


def test_domain_token_target_without_max_uses_profile_limit(model: Settings) -> None:
    with pytest.raises(ValidationError, match="legal_chunk_target_tokens"):
        validate_chunking(
            model,
            {
                ("legal_chunk_target_tokens", "legal"): str(
                    LegalDomainProfile().chunking_policy().max_tokens + 1
                )
            },
        )


def test_domain_token_target_below_profile_max_is_accepted(model: Settings) -> None:
    validate_chunking(
        model,
        {("legal_chunk_target_tokens", "legal"): str(LegalDomainProfile().chunking_policy().max_tokens - 1)},
    )


def test_domain_token_max_below_domain_target_is_rejected(model: Settings) -> None:
    with pytest.raises(ValidationError, match="decree_chunk_target_tokens"):
        validate_chunking(
            model,
            {("decree_chunk_max_tokens", "decree"): "300", ("decree_chunk_target_tokens", "decree"): "301"},
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("embedding_http_max_connections", 0),
        ("rerank_http_max_connections", -1),
        ("embedding_http_max_keepalive_connections", -1),
        ("rerank_http_max_keepalive_connections", 21),
        ("embedding_http_max_connections", 19),
    ],
)
def test_invalid_pool_snapshot_is_rejected_before_mutation(model: Settings, key: str, value: int) -> None:
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus())
    with pytest.raises(ValidationError):
        config.apply([row(key, value)])
    assert getattr(model, key) != value


@pytest.mark.parametrize(
    "key,value",
    [
        ("openrouter_base_url", "https://evil.example/api/v1"),
        ("openrouter_base_url", "http://openrouter.ai/api/v1"),
        ("deepinfra_base_url", "https://api.deepinfra.com.evil.example/v1"),
        ("tei_embed_url", "http://169.254.169.254/latest/meta-data"),
        ("tei_rerank_url", "http://unapproved:8080"),
        ("openai_compatible_base_url", "http://169.254.169.254/v1"),
        ("openai_compatible_base_url", "https://api.openai.com.evil.example/v1"),
        ("tei_embed_url", "file:///etc/passwd"),
        ("tei_embed_url", "http://user:password@tei-embed:8080"),
        ("tei_embed_url", "http://tei-embed:8080/?token=secret"),
    ],
)
def test_untrusted_origins_rejected_without_http_request(model: Settings, key: str, value: str) -> None:
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus())
    with pytest.raises(ValidationError):
        config.validate_change(key, value)


@pytest.mark.parametrize(
    "update,error",
    [
        ({}, "OPENAI_COMPATIBLE_BASE_URL required"),
        ({"openai_compatible_base_url": "http://vllm:8000/v1"}, "OPENAI_COMPATIBLE_MODEL required"),
    ],
)
def test_openai_compatible_requires_endpoint_and_model(
    model: Settings, update: dict[str, str], error: str
) -> None:
    candidate = model.model_copy(
        update={"llm_provider": "openai_compatible", "openrouter_api_key": SecretStr(""), **update}
    )
    with pytest.raises(ValidationError, match=error):
        validate_provider_credentials(candidate)


def test_openai_compatible_needs_no_openrouter_key(model: Settings) -> None:
    candidate = model.model_copy(
        update={
            "llm_provider": "openai_compatible",
            "openrouter_api_key": SecretStr(""),
            "openai_compatible_base_url": "http://vllm:8000/v1",
            "openai_compatible_model": "local",
        }
    )
    validate_provider_credentials(candidate)

    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY required"):
        validate_provider_credentials(candidate.model_copy(update={"llm_provider": "openrouter"}))


def use_factory_settings(monkeypatch: pytest.MonkeyPatch, model: Settings) -> None:
    """Make ML client factories build from ``model`` instead of the process settings."""
    monkeypatch.setattr(factories, "ml_client_settings", lambda: model)
    monkeypatch.setattr(factories, "process_section", lambda _section: model)


def wire_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> list[httpx.AsyncClient]:
    real = httpx.AsyncClient
    pools = []

    def create(**kwargs: Any) -> httpx.AsyncClient:
        result = real(**{**kwargs, "transport": httpx.MockTransport(handler)})
        pools.append(result)
        return result

    monkeypatch.setattr(httpx, "AsyncClient", create)
    return pools


def successful_response(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/info":
        return httpx.Response(
            200,
            json={
                "model_id": "BAAI/bge-m3",
                "model_sha": "same-sha",
                "model_dtype": "float32",
                "model_type": {"embedding": {"pooling": "cls"}},
            },
        )

    if request.url.path == "/embed":
        return httpx.Response(200, json=[[0.25] * 1024])

    if request.url.path == "/rerank":
        return httpx.Response(200, json=[{"index": 0, "score": 0.7}])

    return httpx.Response(200, json={})


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "wrong-model", "wrong-dimension", "unauthorized"])
async def test_cold_bootstrap_probes_default_encoder_before_readiness(
    model: Settings, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    tei_defaults = build_defaults()

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)

        if fault == "unauthorized":
            return httpx.Response(401)

        if fault == "wrong-model" and request.url.path == "/info":
            return httpx.Response(200, json={"model_id": "another-model"})

        if fault == "wrong-dimension" and request.url.path == "/embed":
            return httpx.Response(200, json=[[0.1]])

        return successful_response(request)

    wire_http(monkeypatch, handler)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(
        model, DomainSettingsAdapter(), bus=InProcessEventBus(), ml_clients=registry
    )
    try:
        if fault:
            with pytest.raises((ValidationError, httpx.HTTPError)):
                await config.apply_async(tei_defaults, require_complete=True)
            assert not config.status().ready
            assert config.status().applied_version is None
            assert registry.cached_embeddings is None
        else:
            await config.apply_async(tei_defaults, require_complete=True)
            assert config.status().ready
            assert "/info" in requests

            assert "/embed" in requests
            assert await registry.embeddings().embed_query("query") == [0.25] * 1024
    finally:
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["offline", "dns"])
@pytest.mark.parametrize("path", ["/info", "/embed"])
async def test_cold_bootstrap_warns_when_embedding_endpoint_is_unavailable(
    model: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fault: str, path: str
) -> None:
    tei_defaults = build_defaults()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == path:
            if fault == "dns":
                message = "Temporary failure in name resolution"
                raise httpx.ConnectError(message, request=request)

            return httpx.Response(503)

        return successful_response(request)

    wire_http(monkeypatch, handler)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(
        model, DomainSettingsAdapter(), bus=InProcessEventBus(), ml_clients=registry
    )
    try:
        await config.apply_async(tei_defaults, require_complete=True)
        assert config.status().ready
        assert config.status().applied_version is not None
        assert registry.cached_embeddings is not None
        assert model.ml_provider == "tei"
        assert "endpoint is unavailable during startup" in caplog.text
        assert "corpus migration" in caplog.text
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_switch_keeps_active_request_then_closes_old_pool(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    hosts = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embed" and request.url.host == "tei-embed":
            started.set()
            await release.wait()

        hosts.append(request.url.host)
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    use_factory_settings(monkeypatch, model)
    registry, config = configured_registry(model)
    old = registry.embeddings()
    call = asyncio.create_task(old.embed_query("old call"))
    await started.wait()
    try:
        await config.apply_async([row("tei_embed_url", "http://replacement:8080")])
        assert registry.embeddings() is not old
        assert not any(pool.is_closed for pool in pools[:1])
        assert await registry.embeddings().embed_query("new call") == [0.25] * 1024
        release.set()
        assert await call == [0.25] * 1024
        await cast("TEIEmbeddingsClient", old).close()
        assert pools[0].is_closed
        assert hosts[-1] == "tei-embed"
        await asyncio.sleep(0)
        assert all(client is not old for client in registry._retired_clients)
    finally:
        release.set()
        await call
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["offline", "dns", "wrong-model", "wrong-revision", "wrong-dimension", "redirect"]
)
async def test_failed_preparation_retains_applied_resource(
    model: Settings, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "replacement":
            if failure == "dns":
                message = "Temporary failure in name resolution"
                raise httpx.ConnectError(message, request=request)

            if failure in {"offline", "redirect"}:
                return httpx.Response(
                    503 if failure == "offline" else 302, headers={"Location": "http://evil.example"}
                )

            if request.url.path == "/info" and failure in {"wrong-model", "wrong-revision"}:
                data = successful_response(request).json()
                data["model_id" if failure == "wrong-model" else "model_sha"] = "wrong"
                return httpx.Response(200, json=data)

            if request.url.path == "/embed" and failure == "wrong-dimension":
                return httpx.Response(200, json=[[0.1]])

        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    use_factory_settings(monkeypatch, model)
    registry, config = configured_registry(model)
    old = registry.embeddings()
    try:
        with pytest.raises((ValidationError, httpx.HTTPError)):
            await config.apply_async([row("tei_embed_url", "http://replacement:8080")])
        assert registry.embeddings() is old
        assert not any(pool.is_closed for pool in pools[:1])
        assert model.tei_embed_url == "http://tei-embed:8080"
        assert not config.status().ready
        assert all(pool.is_closed for pool in pools[1:])
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_repeated_pool_changes_close_retired_connections(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pools = wire_http(monkeypatch, successful_response)
    use_factory_settings(monkeypatch, model)
    registry, config = configured_registry(model)
    registry.embeddings()
    try:
        for limit in (41, 42, 43):
            await config.apply_async([row("embedding_http_max_connections", limit)])
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert sum(not pool.is_closed for pool in pools) == 1
            assert not registry._retired_clients
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_failed_reaction_keeps_clients_and_settings(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:

    pools = wire_http(monkeypatch, successful_response)
    use_factory_settings(monkeypatch, model)
    registry = MLClientRegistry()
    bus = InProcessEventBus()
    config = SettingsProviderConfiguration(model, bus=bus, ml_clients=registry)
    old = registry.reranker()

    def fail(event: object) -> None:
        message = "reaction failure"
        raise RuntimeError(message)

    bus.subscribe(ConfigParameterChanged, fail)
    try:
        with pytest.raises(RuntimeError):
            await config.apply_async([row("rerank_http_max_connections", 30)])
        assert model.rerank_http_max_connections == 20
        assert registry.reranker() is old
        assert not any(pool.is_closed for pool in pools[:1])
        assert all(pool.is_closed for pool in pools[1:])
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_stale_probe_cannot_publish_over_newer_database_snapshot(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:

    factory = FakeUnitOfWorkFactory()
    await factory._uow.config_parameters.save(row("tei_rerank_url", "http://replacement:8080"))

    async def handler(request: httpx.Request) -> httpx.Response:
        await factory._uow.config_parameters.update_value("tei_rerank_url", "http://tei-rerank:8080")
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    registry, config = configured_registry(model)
    try:
        with pytest.raises(ValidationError, match="changed during"):
            await load_config_from_db(factory, config, require_complete=False)
        assert model.tei_rerank_url == "http://tei-rerank:8080"
        assert all(pool.is_closed for pool in pools)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_unproven_local_tei_weights_cannot_be_switched(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response = successful_response(request)

        if request.url.path == "/info":
            data = response.json()
            data["model_sha"] = None
            return httpx.Response(200, json=data)

        return response

    wire_http(monkeypatch, handler)
    use_factory_settings(monkeypatch, model)
    registry, config = configured_registry(model)
    registry.embeddings()
    try:
        with pytest.raises(ValidationError, match="revision"):
            await config.apply_async([row("tei_embed_url", "http://replacement:8080")])
        assert model.tei_embed_url == "http://tei-embed:8080"
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_admin_endpoint_failure_never_commits_invalid_setting(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:

    pools = wire_http(monkeypatch, lambda _request: httpx.Response(503))
    registry, config = configured_registry(model)
    factory = FakeUnitOfWorkFactory()
    await factory._uow.config_parameters.save(row("tei_rerank_url", model.tei_rerank_url))
    service = ConfigService(factory, config.bus, config)
    try:
        with pytest.raises((ValidationError, httpx.HTTPError)):
            await service.update_parameter("tei_rerank_url", "http://replacement:8080")
        saved = await factory._uow.config_parameters.get_by_key_and_domain("tei_rerank_url", None)
        assert saved is not None
        assert saved.value == "http://tei-rerank:8080"
        assert all(pool.is_closed for pool in pools)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_admin_probe_has_no_open_transaction_and_detects_concurrent_change(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:

    factory = FakeUnitOfWorkFactory()
    await factory._uow.config_parameters.save(row("tei_rerank_url", model.tei_rerank_url))
    await factory._uow.config_parameters.save(row("rerank_http_max_connections", 20))
    transactions = 0
    original = factory.create

    @asynccontextmanager
    async def transaction(master: bool = False) -> AsyncIterator[FakeUnitOfWork]:
        nonlocal transactions
        transactions += 1
        try:
            async with original(master=master) as uow:
                yield uow
        finally:
            transactions -= 1

    monkeypatch.setattr(factory, "create", transaction)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert transactions == 0
        await factory._uow.config_parameters.update_value("rerank_http_max_connections", "30")
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    registry, config = configured_registry(model)
    try:
        with pytest.raises(ValidationError, match="changed during"):
            await ConfigService(factory, config.bus, config).update_parameter(
                "tei_rerank_url", "http://replacement:8080"
            )
        saved = await factory._uow.config_parameters.get_by_key_and_domain("tei_rerank_url", None)
        assert saved is not None
        assert saved.value == "http://tei-rerank:8080"
        assert all(pool.is_closed for pool in pools)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_health_probe_reads_applied_endpoint(model: Settings, monkeypatch: pytest.MonkeyPatch) -> None:

    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return successful_response(request)

    wire_http(monkeypatch, handler)
    monkeypatch.setattr("config.settings", model)
    registry, config = configured_registry(model)
    try:
        await config.apply_async([row("tei_rerank_url", "http://replacement:8080")])
        hosts.clear()
        await system_health_probe.SystemHealthProbe().check_tei()
        assert "replacement" in hosts
        assert "tei-rerank" not in hosts
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_cached_llm_family_changes_together_and_active_stream_survives(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:

    model.openrouter_base_url = "https://openrouter.ai/old/v1"
    model.ml_internal_allowed_origins += ["https://openrouter.ai/old/v1", "https://openrouter.ai/new/v1"]
    wire_http(monkeypatch, successful_response)
    use_factory_settings(monkeypatch, model)
    registry, started, finish = llm_family(model, monkeypatch)
    old = registry.llm()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    parts = []

    async def consume() -> None:
        parts.extend([part.content async for part in old.astream("question")])

    stream = asyncio.create_task(consume())
    await started.wait()
    try:
        await config.apply_async([row("openrouter_base_url", "https://openrouter.ai/new/v1")])
        assert (await registry.llm().ainvoke("new")).content == "https://openrouter.ai/new/v1"
        assert (await registry.fast_llm().ainvoke("new")).content == "https://openrouter.ai/new/v1"
        assert (
            await registry.llm_for_breadth(Breadth.BROAD.value).ainvoke("new")
        ).content == "https://openrouter.ai/new/v1"
        assert not cast("httpx.AsyncClient", cast("SimpleNamespace", old.client).root_async_client).is_closed
        finish.set()
        await stream
        await old.close()
        assert parts == ["https://openrouter.ai/old/v1", "https://openrouter.ai/old/v1"]
        assert cast("httpx.AsyncClient", cast("SimpleNamespace", old.client).root_async_client).is_closed
    finally:
        finish.set()
        await stream
        await registry.close()


@pytest.mark.asyncio
async def test_upload_finishes_all_embedding_batches_on_original_snapshot(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:

    started, finish = asyncio.Event(), asyncio.Event()
    hosts = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embed":
            hosts.append(request.url.host)

            if len(hosts) == 1:
                started.set()
                await finish.wait()

        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    use_factory_settings(monkeypatch, model)
    monkeypatch.setattr(settings, "embed_batch_size", 1)
    monkeypatch.setattr(qdrant_ops, "ensure_embedding_identity", lambda *_args: "tei:BAAI/bge-m3:v1")
    registry, config = configured_registry(model)
    old = registry.embeddings()
    upload, persisted = start_upload(cast("TEIEmbeddingsClient", old))
    await started.wait()
    try:
        await config.apply_async([row("embedding_http_max_connections", 45)])
        assert not any(pool.is_closed for pool in pools[:1])
        finish.set()
        await upload
        assert len(persisted) == 3
        assert hosts == ["tei-embed"] * 3
        await cast("TEIEmbeddingsClient", old).close()
        assert pools[0].is_closed
    finally:
        finish.set()
        await asyncio.gather(upload, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
async def test_queued_retrieval_acquires_current_embedding_client_after_admission(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:

    queued, release = asyncio.Event(), asyncio.Event()
    requests = []

    class Admission:
        async def __aenter__(self) -> None:
            queued.set()
            await release.wait()

        async def __aexit__(self, *args: object) -> None:
            pass

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    use_factory_settings(monkeypatch, model)
    monkeypatch.setattr(rag_retrieval, "ensure_embedding_identity", lambda *_args: None)
    registry = MLClientRegistry()
    monkeypatch.setattr(registry.corpus_resources, "client", SimpleNamespace(search=lambda **_kwargs: []))
    monkeypatch.setattr(registry, "embedding_semaphore", Admission())
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    search = asyncio.create_task(rag_retrieval.qdrant_dense_search("query", 1, Filter(), registry))
    await queued.wait()
    try:
        await config.apply_async([row("embedding_http_max_connections", 45)])
        await asyncio.sleep(0)
        release.set()
        assert await search == []
        assert len(requests) == 1
        assert pools[0].is_closed
    finally:
        release.set()
        await asyncio.gather(search, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
async def test_auxiliary_call_waiting_for_semaphore_keeps_its_original_client(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:

    queued, release = asyncio.Event(), asyncio.Event()

    class Admission:
        async def __aenter__(self) -> None:
            queued.set()
            await release.wait()

        async def __aexit__(self, *args: object) -> None:
            pass

    wire_http(monkeypatch, successful_response)
    use_factory_settings(monkeypatch, model)

    def create(model: Settings | None = None, *, runtime: object = None) -> SimpleNamespace:
        async def invoke(*args: object, **kwargs: Any) -> AIMessage:
            return AIMessage(content="expanded question")

        return SimpleNamespace(ainvoke=invoke, root_async_client=httpx.AsyncClient(), root_client=None)

    monkeypatch.setattr(factories, "create_fast_llm_for_auxiliary", create)
    registry = MLClientRegistry()
    monkeypatch.setattr(registry, "auxiliary_semaphore", Admission())
    old = registry.fast_llm()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    call = asyncio.create_task(
        condense_question(old, "question", [HumanMessage(content="previous")], registry)
    )
    await queued.wait()
    try:
        await config.apply_async([ConfigParameter("llm_temperature", "0.2", ConfigValueType.FLOAT, "llm")])
        assert not cast("httpx.AsyncClient", cast("SimpleNamespace", old.client).root_async_client).is_closed
        release.set()
        assert await call == "expanded question"
        await old.close()
        assert cast("httpx.AsyncClient", cast("SimpleNamespace", old.client).root_async_client).is_closed
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
async def test_promoted_migration_rows_show_current_metadata_and_preserve_database_value() -> None:

    factory = FakeUnitOfWorkFactory()
    saved = ConfigParameter(
        "rerank_http_max_connections",
        "26",
        ConfigValueType.INT,
        "migration_pending",
        description="Pending implementation; env retained",
    )
    await factory._uow.config_parameters.save(saved)
    listed = await ConfigService(factory, InProcessEventBus()).list_parameters()
    assert listed[0].category == "ml"
    assert listed[0].description == "Reranker HTTP connections per process"
    assert listed[0].min_value == 1
    assert listed[0].value == saved.value == "26"


def test_factory_captures_one_snapshot_even_if_settings_change_during_construction(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_limits = httpx.Limits

    def limits(**kwargs: Any) -> httpx.Limits:
        model.tei_embed_url = "http://replacement:8080"
        model.tei_embed_model = "different-model"
        return real_limits(**kwargs)

    monkeypatch.setattr(httpx, "Limits", limits)
    client = factories.create_embeddings(model)
    try:
        assert client._base_url == "http://tei-embed:8080"
        assert client.embedding_identity == "tei:BAAI/bge-m3:v2"
    finally:
        asyncio.run(client.close())


def configured_registry(model: Settings) -> tuple[MLClientRegistry, SettingsProviderConfiguration]:
    registry = MLClientRegistry()
    return registry, SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)


def llm_family(
    model: Settings, monkeypatch: pytest.MonkeyPatch
) -> tuple[MLClientRegistry, asyncio.Event, asyncio.Event]:
    started, finish = asyncio.Event(), asyncio.Event()
    created = []

    class LLM:
        def __init__(self, endpoint: str) -> None:
            self.endpoint = endpoint
            self.root_async_client = httpx.AsyncClient()
            self.root_client = None
            created.append(self)

        async def ainvoke(self, *_args: object, **_kwargs: Any) -> AIMessage:
            return AIMessage(content=self.endpoint)

        async def astream(self, *_args: object, **_kwargs: Any) -> AsyncIterator[AIMessageChunk]:
            started.set()
            yield AIMessageChunk(content=self.endpoint)
            await finish.wait()
            yield AIMessageChunk(content=self.endpoint)

    initial_settings = model

    def llm(*_args: object, model: Settings | None = None, **_kwargs: Any) -> LLM:
        return LLM((model or initial_settings).openrouter_base_url)

    monkeypatch.setattr(factories, "create_llm", llm)
    monkeypatch.setattr(factories, "create_fast_llm_for_auxiliary", llm)
    monkeypatch.setattr(factories, "create_llm_for_breadth", llm)
    registry = MLClientRegistry()
    registry.llm()
    registry.fast_llm()
    registry.llm_for_breadth(Breadth.BROAD.value)
    return registry, started, finish


def start_upload(old: TEIEmbeddingsClient) -> tuple[asyncio.Task[None], list[PointStruct]]:
    class Qdrant:
        def upsert(self, collection_name: str, points: list[PointStruct]) -> None:
            assert collection_name
            persisted.extend(points)

    persisted: list[PointStruct] = []
    upload = asyncio.create_task(
        qdrant_ops.upload_to_qdrant(
            [Document(page_content=f"chunk {i}", metadata={"chunk_id": i}) for i in range(3)],
            old,
            client=MagicMock(wraps=Qdrant()),
        )
    )
    return upload, persisted


def chunk_row(key: str, value: int, domain_key: str | None = None) -> ConfigParameter:
    return ConfigParameter(key, str(value), ConfigValueType.INT, "ingestion", domain_key=domain_key)


def test_global_chunk_overlap_must_stay_below_size(model: Settings) -> None:
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus())

    with pytest.raises(ValidationError, match="CHUNK_OVERLAP must be less than CHUNK_SIZE"):
        config.apply([chunk_row("chunk_size", 150)])  # default overlap is 200
    assert model.chunk_size != 150

    config.apply([chunk_row("chunk_size", 400), chunk_row("chunk_overlap", 100)])
    assert (model.chunk_size, model.chunk_overlap) == (400, 100)


def int_row(key: str, value: int, category: str) -> ConfigParameter:
    return ConfigParameter(key, str(value), ConfigValueType.INT, category)


def test_snapshot_rejects_top_k_above_fetch_k(model: Settings) -> None:
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus())

    with pytest.raises(ValidationError, match="retriever_top_k must not exceed retriever_fetch_k"):
        config.apply([int_row("retriever_top_k", 50, "rag"), int_row("retriever_fetch_k", 1, "rag")])
    assert (model.retriever_top_k, model.retriever_fetch_k) != (50, 1)

    config.apply([int_row("retriever_top_k", 10, "rag"), int_row("retriever_fetch_k", 30, "rag")])
    assert (model.retriever_top_k, model.retriever_fetch_k) == (10, 30)


def test_snapshot_rejects_num_predict_not_below_num_ctx(model: Settings) -> None:
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus())

    with pytest.raises(ValidationError, match="llm_num_predict_broad must be less than llm_num_ctx_broad"):
        config.apply(
            [int_row("llm_num_ctx_broad", 2048, "llm"), int_row("llm_num_predict_broad", 2048, "llm")]
        )
    assert model.llm_num_ctx_broad != 2048

    config.apply([int_row("llm_num_ctx_broad", 2048, "llm"), int_row("llm_num_predict_broad", 1024, "llm")])
    assert (model.llm_num_ctx_broad, model.llm_num_predict_broad) == (2048, 1024)


def test_domain_token_target_must_stay_within_max(model: Settings) -> None:
    domain_settings = DomainSettingsAdapter()
    domain_settings.replace_snapshot(
        {("legal_chunk_max_tokens", "legal"): "500", ("legal_chunk_target_tokens", "legal"): "50"}
    )
    config = SettingsProviderConfiguration(model, domain_settings, bus=InProcessEventBus())

    with pytest.raises(
        ValidationError, match="legal_chunk_target_tokens must be positive and <= legal_chunk_max_tokens"
    ):
        config.apply([chunk_row("legal_chunk_target_tokens", 501, "legal")])
    assert domain_settings.get("legal_chunk_target_tokens", "legal") == "50"

    config.apply([chunk_row("legal_chunk_target_tokens", 120, "legal")])
    assert domain_settings.get("legal_chunk_target_tokens", "legal") == "120"
