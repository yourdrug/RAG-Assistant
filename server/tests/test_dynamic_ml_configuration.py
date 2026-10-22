"""ML snapshot switching at the real HTTP and resource ownership boundaries."""

import asyncio

import httpx
import pytest

from config import Settings
from domain.runtime.config import ConfigParameter
from domain.shared.exceptions import ValidationError
from domain.runtime.config import ConfigValueType
from domain.runtime.dynamic_params import DYNAMIC_FIELDS_MAP
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.ml.clients import factories
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from infrastructure.initialization import build_defaults
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter


ENDPOINT_KEYS = (
    "ollama_base_url",
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


def row(key, value):
    return ConfigParameter(
        key, str(value), ConfigValueType.INT if key in POOL_KEYS else ConfigValueType.STR, "ml"
    )


@pytest.fixture
def model():
    return Settings(
        _env_file=None,
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
def test_keys_have_working_dynamic_declarations(key):
    assert key in DYNAMIC_FIELDS_MAP


@pytest.mark.parametrize("prefix, connections, keepalive", [("embedding", 40, 20), ("rerank", 20, 10)])
def test_factories_use_independent_effective_pool_limits(model, prefix, connections, keepalive, monkeypatch):
    captured = []
    real = httpx.AsyncClient

    def client(**kwargs):
        captured.append(kwargs["limits"])
        return real(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    adapter = (
        factories.create_embeddings(model) if prefix == "embedding" else factories.create_reranker(model)
    )
    assert captured[0].max_connections == connections
    assert captured[0].max_keepalive_connections == keepalive
    asyncio.run(adapter.close())


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
def test_invalid_pool_snapshot_is_rejected_before_mutation(model, key, value):
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
        ("ollama_base_url", "file:///etc/passwd"),
        ("ollama_base_url", "http://user:password@ollama:11434"),
        ("ollama_base_url", "http://ollama:11434/?token=secret"),
    ],
)
def test_untrusted_origins_rejected_without_http_request(model, key, value):
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus())
    with pytest.raises(ValidationError):
        config.validate_change(key, value)


def wire_http(monkeypatch, handler):
    real = httpx.AsyncClient
    pools = []

    def create(**kwargs):
        result = real(**{**kwargs, "transport": httpx.MockTransport(handler)})
        pools.append(result)
        return result

    monkeypatch.setattr(httpx, "AsyncClient", create)
    return pools


def successful_response(request):
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
@pytest.mark.parametrize('fault', [None, 'wrong-model', 'wrong-dimension', 'offline'])
async def test_cold_bootstrap_probes_default_encoder_before_readiness(model, monkeypatch, fault):
    requests = []

    def handler(request):
        requests.append(request.url.path)
        if fault == 'offline':
            return httpx.Response(503)
        if fault == 'wrong-model' and request.url.path == '/info':
            return httpx.Response(200, json={'model_id': 'another-model'})
        if fault == 'wrong-dimension' and request.url.path == '/embed':
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
                await config.apply_async(build_defaults(), require_complete=True)
            assert not config.status().ready
            assert config.status().applied_version is None
            assert registry._embeddings is None
        else:
            await config.apply_async(build_defaults(), require_complete=True)
            assert config.status().ready
            assert '/info' in requests and '/embed' in requests
            assert await registry.embeddings().embed_query('query') == [0.25] * 1024
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_switch_keeps_active_request_then_closes_old_pool(model, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    hosts = []

    async def handler(request):
        if request.url.path == "/embed" and request.url.host == "tei-embed":
            started.set()
            await release.wait()
        hosts.append(request.url.host)
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    monkeypatch.setattr(factories, "settings", model)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    old = registry.embeddings()
    call = asyncio.create_task(old.embed_query("old call"))
    await started.wait()
    try:
        await config.apply_async([row("tei_embed_url", "http://replacement:8080")])
        assert registry.embeddings() is not old
        assert not pools[0].is_closed
        assert await registry.embeddings().embed_query("new call") == [0.25] * 1024
        release.set()
        assert await call == [0.25] * 1024
        await old.close()
        assert pools[0].is_closed
        assert hosts[-1] == "tei-embed"
        await asyncio.sleep(0)
        assert old not in registry._retired_clients
    finally:
        release.set()
        await call
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["offline", "wrong-model", "wrong-revision", "wrong-dimension", "redirect"]
)
async def test_failed_preparation_retains_applied_resource(model, monkeypatch, failure):
    def handler(request):
        if request.url.host == "replacement":
            if failure == "offline":
                return httpx.Response(503)
            if failure == "redirect":
                return httpx.Response(302, headers={"Location": "http://evil.example"})
            if request.url.path == "/info" and failure in {"wrong-model", "wrong-revision"}:
                data = successful_response(request).json()
                data["model_id" if failure == "wrong-model" else "model_sha"] = "wrong"
                return httpx.Response(200, json=data)
            if request.url.path == "/embed" and failure == "wrong-dimension":
                return httpx.Response(200, json=[[0.1]])
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    monkeypatch.setattr(factories, "settings", model)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    old = registry.embeddings()
    try:
        with pytest.raises((ValidationError, httpx.HTTPError)):
            await config.apply_async([row("tei_embed_url", "http://replacement:8080")])
        assert registry.embeddings() is old
        assert not pools[0].is_closed
        assert model.tei_embed_url == "http://tei-embed:8080"
        assert not config.status().ready
        assert all(pool.is_closed for pool in pools[1:])
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_repeated_pool_changes_close_retired_connections(model, monkeypatch):
    pools = wire_http(monkeypatch, successful_response)
    monkeypatch.setattr(factories, "settings", model)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
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
async def test_failed_reaction_keeps_clients_and_settings(model, monkeypatch):
    from domain.runtime.config import ConfigParameterChanged

    pools = wire_http(monkeypatch, successful_response)
    monkeypatch.setattr(factories, "settings", model)
    registry = MLClientRegistry()
    bus = InProcessEventBus()
    config = SettingsProviderConfiguration(model, bus=bus, ml_clients=registry)
    old = registry.reranker()

    def fail(event):
        raise RuntimeError("reaction failure")

    bus.subscribe(ConfigParameterChanged, fail)
    try:
        with pytest.raises(RuntimeError):
            await config.apply_async([row("rerank_http_max_connections", 30)])
        assert model.rerank_http_max_connections == 20
        assert registry.reranker() is old
        assert not pools[0].is_closed
        assert all(pool.is_closed for pool in pools[1:])
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_stale_probe_cannot_publish_over_newer_database_snapshot(model, monkeypatch):
    from fakes import FakeUnitOfWorkFactory
    from infrastructure.initialization import load_config_from_db

    factory = FakeUnitOfWorkFactory()
    await factory._uow.config_parameters.save(row("tei_rerank_url", "http://replacement:8080"))

    async def handler(request):
        await factory._uow.config_parameters.update_value("tei_rerank_url", "http://tei-rerank:8080")
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    try:
        with pytest.raises(ValidationError, match="changed during"):
            await load_config_from_db(factory, config, require_complete=False)
        assert model.tei_rerank_url == "http://tei-rerank:8080"
        assert all(pool.is_closed for pool in pools)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_unproven_local_tei_weights_cannot_be_switched(model, monkeypatch):
    def handler(request):
        response = successful_response(request)
        if request.url.path == "/info":
            data = response.json()
            data["model_sha"] = None
            return httpx.Response(200, json=data)
        return response

    wire_http(monkeypatch, handler)
    monkeypatch.setattr(factories, "settings", model)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    registry.embeddings()
    try:
        with pytest.raises(ValidationError, match="revision"):
            await config.apply_async([row("tei_embed_url", "http://replacement:8080")])
        assert model.tei_embed_url == "http://tei-embed:8080"
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_admin_endpoint_failure_never_commits_invalid_setting(model, monkeypatch):
    from application.admin import ConfigService
    from fakes import FakeUnitOfWorkFactory

    pools = wire_http(monkeypatch, lambda request: httpx.Response(503))
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    factory = FakeUnitOfWorkFactory()
    await factory._uow.config_parameters.save(row("tei_rerank_url", model.tei_rerank_url))
    service = ConfigService(factory, config.bus, config)
    try:
        with pytest.raises((ValidationError, httpx.HTTPError)):
            await service.update_parameter("tei_rerank_url", "http://replacement:8080")
        saved = await factory._uow.config_parameters.get_by_key_and_domain("tei_rerank_url", None)
        assert saved.value == "http://tei-rerank:8080"
        assert all(pool.is_closed for pool in pools)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_admin_probe_has_no_open_transaction_and_detects_concurrent_change(model, monkeypatch):
    from contextlib import asynccontextmanager
    from application.admin import ConfigService
    from fakes import FakeUnitOfWorkFactory

    factory = FakeUnitOfWorkFactory()
    await factory._uow.config_parameters.save(row("tei_rerank_url", model.tei_rerank_url))
    await factory._uow.config_parameters.save(row("rerank_http_max_connections", 20))
    transactions = 0
    original = factory.create

    @asynccontextmanager
    async def transaction(master=False):
        nonlocal transactions
        transactions += 1
        try:
            async with original(master=master) as uow:
                yield uow
        finally:
            transactions -= 1

    monkeypatch.setattr(factory, "create", transaction)

    async def handler(request):
        assert transactions == 0
        await factory._uow.config_parameters.update_value("rerank_http_max_connections", "30")
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    try:
        with pytest.raises(ValidationError, match="changed during"):
            await ConfigService(factory, config.bus, config).update_parameter(
                "tei_rerank_url", "http://replacement:8080"
            )
        saved = await factory._uow.config_parameters.get_by_key_and_domain("tei_rerank_url", None)
        assert saved.value == "http://tei-rerank:8080"
        assert all(pool.is_closed for pool in pools)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_health_probe_reads_applied_endpoint(model, monkeypatch):
    from infrastructure.health import system_health_probe

    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        return successful_response(request)

    wire_http(monkeypatch, handler)
    monkeypatch.setattr(system_health_probe, "settings", model)
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    try:
        await config.apply_async([row("tei_rerank_url", "http://replacement:8080")])
        hosts.clear()
        await system_health_probe.SystemHealthProbe().check_tei()
        assert "replacement" in hosts
        assert "tei-rerank" not in hosts
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_cached_llm_family_changes_together_and_active_stream_survives(model, monkeypatch):
    from domain.rag.breadth import Breadth

    model.ollama_base_url = "http://old:11434"
    model.ml_internal_allowed_origins += ["http://old:11434", "http://new:11434"]
    wire_http(monkeypatch, successful_response)
    monkeypatch.setattr(factories, "settings", model)
    started, finish = asyncio.Event(), asyncio.Event()
    created = []

    class LLM:
        def __init__(self, endpoint):
            self.endpoint = endpoint
            self.root_async_client = httpx.AsyncClient()
            self.root_client = None
            created.append(self)

        async def ainvoke(self, *args, **kwargs):
            return self.endpoint

        async def astream(self, *args, **kwargs):
            started.set()
            yield self.endpoint
            await finish.wait()
            yield self.endpoint

    def llm(*args, model=None, **kwargs):
        return LLM((model or factories.settings).ollama_base_url)

    monkeypatch.setattr(factories, "create_llm", llm)
    monkeypatch.setattr(factories, "create_fast_llm_for_auxiliary", llm)
    monkeypatch.setattr(factories, "create_llm_for_breadth", llm)
    registry = MLClientRegistry()
    old = registry.llm()
    registry.fast_llm()
    registry.llm_for_breadth(Breadth.BROAD.value)
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    parts = []

    async def consume():
        async for part in old.astream("question"):
            parts.append(part)

    stream = asyncio.create_task(consume())
    await started.wait()
    try:
        await config.apply_async([row("ollama_base_url", "http://new:11434")])
        assert await registry.llm().ainvoke("new") == "http://new:11434"
        assert await registry.fast_llm().ainvoke("new") == "http://new:11434"
        assert await registry.llm_for_breadth(Breadth.BROAD.value).ainvoke("new") == "http://new:11434"
        assert not old.client.root_async_client.is_closed
        finish.set()
        await stream
        await old.close()
        assert parts == ["http://old:11434", "http://old:11434"]
        assert old.client.root_async_client.is_closed
    finally:
        finish.set()
        await stream
        await registry.close()




@pytest.mark.asyncio
async def test_upload_finishes_all_embedding_batches_on_original_snapshot(model, monkeypatch):
    from config import settings
    from langchain_core.documents import Document
    from infrastructure.repositories.vector import qdrant_ops

    started, finish = asyncio.Event(), asyncio.Event()
    hosts = []

    async def handler(request):
        if request.url.path == "/embed":
            hosts.append(request.url.host)
            if len(hosts) == 1:
                started.set()
                await finish.wait()
        return successful_response(request)

    class Qdrant:
        def upsert(self, collection_name, points):
            for point in points:
                persisted.append(point)

    persisted = []
    pools = wire_http(monkeypatch, handler)
    monkeypatch.setattr(factories, "settings", model)
    monkeypatch.setattr(settings, "embed_batch_size", 1)
    monkeypatch.setattr(qdrant_ops, "ensure_embedding_identity", lambda *args: "tei:BAAI/bge-m3:v1")
    registry = MLClientRegistry()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    old = registry.embeddings()
    upload = asyncio.create_task(
        qdrant_ops.upload_to_qdrant(
            [Document(page_content=f"chunk {i}", metadata={"chunk_id": i}) for i in range(3)],
            old,
            client=Qdrant(),
        )
    )
    await started.wait()
    try:
        await config.apply_async([row("embedding_http_max_connections", 45)])
        assert not pools[0].is_closed
        finish.set()
        await upload
        assert len(persisted) == 3
        assert hosts == ["tei-embed"] * 3
        await old.close()
        assert pools[0].is_closed
    finally:
        finish.set()
        await asyncio.gather(upload, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
async def test_queued_retrieval_acquires_current_embedding_client_after_admission(model, monkeypatch):
    from types import SimpleNamespace
    from qdrant_client.models import Filter
    from infrastructure.ml.rag import rag_retrieval

    queued, release = asyncio.Event(), asyncio.Event()
    requests = []

    class Admission:
        async def __aenter__(self):
            queued.set()
            await release.wait()

        async def __aexit__(self, *args):
            pass

    def handler(request):
        requests.append(request)
        return successful_response(request)

    pools = wire_http(monkeypatch, handler)
    monkeypatch.setattr(factories, "settings", model)
    monkeypatch.setattr(rag_retrieval, "ensure_embedding_identity", lambda *args: None)
    registry = MLClientRegistry()
    registry._qdrant_client = SimpleNamespace(search=lambda **kwargs: [])
    registry.embedding_semaphore = Admission()
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
async def test_auxiliary_call_waiting_for_semaphore_keeps_its_original_client(model, monkeypatch):
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage, HumanMessage
    from infrastructure.ml.rag.rag_prompts import condense_question

    queued, release = asyncio.Event(), asyncio.Event()

    class Admission:
        async def __aenter__(self):
            queued.set()
            await release.wait()

        async def __aexit__(self, *args):
            pass

    wire_http(monkeypatch, successful_response)
    monkeypatch.setattr(factories, "settings", model)

    def create(model=None):
        async def invoke(*args, **kwargs):
            return AIMessage(content="expanded question")

        return SimpleNamespace(ainvoke=invoke, root_async_client=httpx.AsyncClient(), root_client=None)

    monkeypatch.setattr(factories, "create_fast_llm_for_auxiliary", create)
    registry = MLClientRegistry()
    registry.auxiliary_semaphore = Admission()
    old = registry.fast_llm()
    config = SettingsProviderConfiguration(model, bus=InProcessEventBus(), ml_clients=registry)
    call = asyncio.create_task(
        condense_question(old, "question", [HumanMessage(content="previous")], registry)
    )
    await queued.wait()
    try:
        await config.apply_async([ConfigParameter("llm_temperature", "0.2", ConfigValueType.FLOAT, "llm")])
        assert not old.client.root_async_client.is_closed
        release.set()
        assert await call == "expanded question"
        await old.close()
        assert old.client.root_async_client.is_closed
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        await registry.close()




@pytest.mark.asyncio
async def test_promoted_migration_rows_show_current_metadata_and_preserve_database_value():
    from application.admin import ConfigService
    from fakes import FakeUnitOfWorkFactory

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


def test_factory_captures_one_snapshot_even_if_settings_change_during_construction(model, monkeypatch):
    real_limits = httpx.Limits

    def limits(**kwargs):
        model.tei_embed_url = "http://replacement:8080"
        model.tei_embed_model = "different-model"
        return real_limits(**kwargs)

    monkeypatch.setattr(httpx, "Limits", limits)
    client = factories.create_embeddings(model)
    try:
        assert client._base_url == "http://tei-embed:8080"
        assert client.embedding_identity == "tei:BAAI/bge-m3:v1"
    finally:
        asyncio.run(client.close())
