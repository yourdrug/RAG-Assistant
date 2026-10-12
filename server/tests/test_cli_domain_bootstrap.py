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


@pytest.mark.asyncio
async def test_cli_loads_domain_settings_before_auto_classification(monkeypatch):
    adapter = DomainSettingsAdapter()
    registry = DomainProfileRegistry()
    register_all_profiles(registry, adapter)
    infra = InfrastructureContainer(domain_settings=adapter, domain_registry=registry)
    infra.init = lambda database: setattr(infra, "_initialized", True)
    bus = InProcessEventBus()
    monkeypatch.setattr("composition.infrastructure.event_bus", bus, raising=False)
    row = SimpleNamespace(key="fingerprint_min_articles", domain_key="legal", value="7", normalize=str)
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    rows = [
        SimpleNamespace(key=default.key, domain_key=profile.key, value=default.value, normalize=str)
        for profile in registry.all()
        for default in profile.config_defaults()
        if (default.key, profile.key) != (row.key, row.domain_key)
    ]
    uow.config_parameters.get_all.return_value = [*rows, row]
    infra.db.uow_factory = SimpleNamespace(create=lambda **kwargs: uow)
    monkeypatch.setattr("composition.infrastructure.seed_domain_config_defaults", AsyncMock(), raising=False)
    monkeypatch.setattr("composition.infrastructure.InfrastructureContainer", lambda: infra)
    monkeypatch.setattr(ingest, "InfrastructureContainer", lambda: infra, raising=False)
    monkeypatch.setattr(database, "connect", AsyncMock())
    monkeypatch.setattr(database, "disconnect", AsyncMock())
    monkeypatch.setattr(redis_client, "init", AsyncMock())
    monkeypatch.setattr(redis_client, "aclose", AsyncMock())

    async def classify(infrastructure):
        assert infrastructure.domain_settings.get("fingerprint_min_articles", "legal") == "7"
        infrastructure.domain_registry.classify(
            "An ordinary document without legal articles.", settings=adapter
        )
        bus.publish(
            ConfigParameterChanged(
                key=row.key,
                domain_key=row.domain_key,
                old_value="7",
                new_value="8",
                value_type=ConfigValueType.INT,
            )
        )
        return infrastructure.domain_settings.get(row.key, row.domain_key)

    result = await ingest.run_with_infrastructure(classify)
    assert result == "8"
    bus.publish(
        ConfigParameterChanged(
            key=row.key,
            domain_key=row.domain_key,
            old_value="8",
            new_value="9",
            value_type=ConfigValueType.INT,
        )
    )
    assert adapter.get(row.key, row.domain_key) == "8"


@pytest.mark.asyncio
async def test_domain_bootstrap_is_idempotent_and_releases_only_owned_handler(monkeypatch):
    adapter = DomainSettingsAdapter()
    infra = InfrastructureContainer(domain_settings=adapter, domain_registry=DomainProfileRegistry())
    bus = InProcessEventBus()
    observed = []
    bus.subscribe(ConfigParameterChanged, observed.append)
    monkeypatch.setattr("composition.infrastructure.event_bus", bus)
    monkeypatch.setattr("composition.infrastructure.seed_domain_config_defaults", AsyncMock())
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.config_parameters.get_all.return_value = []
    infra.db.uow_factory = SimpleNamespace(create=lambda **kwargs: uow)
    await infra.bootstrap_domain_settings()
    await infra.bootstrap_domain_settings()
    assert len(bus._handlers[ConfigParameterChanged]) == 2
    await infra.dispose()
    event = ConfigParameterChanged(
        key="fingerprint_min_articles",
        domain_key="legal",
        old_value=None,
        new_value="5",
        value_type=ConfigValueType.INT,
    )
    bus.publish(event)
    assert observed == [event]
    with pytest.raises(KeyError):
        adapter.get(event.key, event.domain_key)


@pytest.mark.asyncio
async def test_domain_bootstrap_propagates_database_failure_and_can_release_subscription(monkeypatch):
    infra = InfrastructureContainer(
        domain_settings=DomainSettingsAdapter(), domain_registry=DomainProfileRegistry()
    )
    bus = InProcessEventBus()
    monkeypatch.setattr("composition.infrastructure.event_bus", bus)
    monkeypatch.setattr("composition.infrastructure.seed_domain_config_defaults", AsyncMock())
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.config_parameters.get_all.side_effect = RuntimeError("database unavailable")
    infra.db.uow_factory = SimpleNamespace(create=lambda **kwargs: uow)
    with pytest.raises(RuntimeError, match="database unavailable"):
        await infra.bootstrap_domain_settings()
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
