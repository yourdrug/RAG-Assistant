"""Fresh CLI processes must load domain config before auto classification."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from composition.infrastructure import InfrastructureContainer
from domain.domain_profile.registry import DomainProfileRegistry
from domain.events.config_events import ConfigParameterChanged
from domain.value_objects.config_value_type import ConfigValueType
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.events.in_process_event_bus import InProcessEventBus
from presentation.cli.commands import ingest


def build_infrastructure(monkeypatch):
    from config import Settings
    from fakes import FakeUnitOfWorkFactory
    from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
    from infrastructure.events.postgres_config_listener import PostgresConfigListener

    adapter = DomainSettingsAdapter()
    registry = DomainProfileRegistry()
    register_all_profiles(registry, adapter)
    bus = InProcessEventBus()
    monkeypatch.setattr("composition.infrastructure.event_bus", bus)
    infra = InfrastructureContainer(domain_settings=adapter, domain_registry=registry)
    infra.corpus_control = AsyncMock()
    infra.db.uow_factory = FakeUnitOfWorkFactory()
    model = Settings(
        _env_file=None, tei_embed_url="http://tei-embed:80", tei_rerank_url="http://tei-rerank:80"
    )
    infra.configuration = SettingsProviderConfiguration(model, adapter, registry, bus)
    infra.events.config_listener = PostgresConfigListener(
        bus, infra.db.uow, configuration=infra.configuration
    )
    infra.ml.ml_clients = SimpleNamespace(
        invalidate_llm=lambda: None,
        invalidate_bm25=lambda: None,
        invalidate_embeddings=lambda: None,
        invalidate_reranker=lambda: None,
        close=AsyncMock(),
    )
    infra.subscribe_config_events()
    return infra, bus


@pytest.mark.asyncio
async def test_cli_loads_global_and_domain_settings_before_auto_classification(monkeypatch):
    from domain.entities.config_parameter import ConfigParameter

    infra, bus = build_infrastructure(monkeypatch)
    infra.init = lambda database: setattr(infra, "_initialized", True)
    await infra.db.uow_factory._uow.config_parameters.save(
        ConfigParameter(
            "fingerprint_min_articles",
            "7",
            "int",
            "domain:legal",
            domain_key="legal",
        )
    )
    await infra.db.uow_factory._uow.config_parameters.save(
        ConfigParameter("llm_model", "db-model", "str", "llm")
    )
    monkeypatch.setattr(ingest, "InfrastructureContainer", lambda: infra)
    monkeypatch.setattr(database, "connect", AsyncMock())
    monkeypatch.setattr(database, "disconnect", AsyncMock())
    monkeypatch.setattr(redis_client, "init", AsyncMock())
    monkeypatch.setattr(redis_client, "aclose", AsyncMock())

    async def classify(infrastructure):
        assert infrastructure.configuration.settings.llm_model == "db-model"
        assert infrastructure.domain_settings.get("fingerprint_min_articles", "legal") == "7"
        infrastructure.domain_registry.classify("An ordinary document.", settings=infra.domain_settings)
        await infra.db.uow_factory._uow.config_parameters.update_value(
            "fingerprint_min_articles", "8", "legal"
        )
        bus.publish(
            ConfigParameterChanged(
                key="fingerprint_min_articles",
                domain_key="legal",
                old_value="7",
                new_value="8",
                value_type=ConfigValueType.INT,
            )
        )
        await infra.events.config_listener.resync()
        return infrastructure.domain_settings.get("fingerprint_min_articles", "legal")

    assert await ingest.run_with_infrastructure(classify) == "8"
    bus.publish(
        ConfigParameterChanged(
            key="fingerprint_min_articles",
            domain_key="legal",
            old_value="8",
            new_value="9",
            value_type=ConfigValueType.INT,
        )
    )
    assert infra.domain_settings.get("fingerprint_min_articles", "legal") == "8"


@pytest.mark.asyncio
async def test_configuration_bootstrap_is_idempotent_and_releases_only_owned_handlers(monkeypatch):
    infra, bus = build_infrastructure(monkeypatch)
    observed = []
    bus.subscribe(ConfigParameterChanged, observed.append)
    await infra.bootstrap_configuration()
    await infra.bootstrap_configuration()
    assert len(bus._handlers[ConfigParameterChanged]) == 6
    await infra.dispose()
    observed.clear()
    event = ConfigParameterChanged(
        key="fingerprint_min_articles",
        domain_key="legal",
        old_value=None,
        new_value="5",
        value_type=ConfigValueType.INT,
    )
    bus.publish(event)
    assert observed == [event]
    assert infra.domain_settings.get(event.key, event.domain_key) != "5"


@pytest.mark.asyncio
async def test_configuration_bootstrap_propagates_database_failure_and_releases_subscription(monkeypatch):
    infra, bus = build_infrastructure(monkeypatch)
    infra.db.uow_factory._uow.config_parameters.get_all = AsyncMock(
        side_effect=RuntimeError("database unavailable")
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        await infra.bootstrap_configuration()
    await infra.dispose()
    assert not bus._handlers[ConfigParameterChanged]


@pytest.mark.parametrize("fails", [False, True])
def test_cli_upload_closes_container_storage_before_loop_exit(monkeypatch, tmp_path, fails):
    path = tmp_path / "report.txt"
    path.write_bytes(b"content")
    events = []
    storage = SimpleNamespace(upload_file=AsyncMock(), aclose=AsyncMock())
    if fails:
        storage.upload_file.side_effect = OSError("storage unavailable")

    async def run(action):
        try:
            return await action(SimpleNamespace(ml=SimpleNamespace(storage=storage)))
        finally:
            await storage.aclose()
            events.append("closed")

    monkeypatch.setattr(ingest, "run_with_infrastructure", run)
    if fails:
        with pytest.raises(SystemExit) as caught:
            ingest.ingest_upload(str(path), key="docs/report.txt")
        assert caught.value.code == 1
    else:
        ingest.ingest_upload(str(path), key="docs/report.txt")
    storage.upload_file.assert_awaited_once_with("docs/report.txt", b"content")
    storage.aclose.assert_awaited_once()
    assert events == ["closed"]
