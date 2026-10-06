"""Regressions for ML invalidation and resource ownership."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from composition.container import Container
from config import settings
from domain.events.config_events import ConfigParameterChanged
from domain.value_objects.config_value_type import ConfigValueType
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client
from presentation.cli.commands.worker import on_shutdown


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["llm_num_ctx_broad", "llm_num_predict_broad"])
async def test_broad_config_rebuilds_cached_llm(monkeypatch, key):
    bus = InProcessEventBus()
    monkeypatch.setattr("infrastructure.events.in_process_event_bus.event_bus", bus)
    monkeypatch.setattr(settings, key, 16384)

    def create_client(breadth):
        return SimpleNamespace(limit=getattr(settings, key), close=AsyncMock())

    monkeypatch.setattr("infrastructure.ml.clients.factories.create_llm_for_breadth", create_client)
    registry = MLClientRegistry()
    container = Container()
    container.infrastructure.ml.ml_clients = registry
    container.subscribe_config_events()
    try:
        cached = registry.llm_for_breadth("broad")
        assert cached.limit == 16384
        bus.publish(
            ConfigParameterChanged(
                key=key,
                old_value="16384",
                new_value="16896",
                value_type=ConfigValueType.INT.value,
            )
        )
        current = registry.llm_for_breadth("broad")
        assert current is not cached
        assert current.limit == getattr(settings, key) == 16896
    finally:
        await container.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "application", "events", "storage"])
async def test_container_closes_ml_registry_even_if_other_cleanup_fails(failure):
    container = Container()
    registry = MLClientRegistry()
    embedding = SimpleNamespace(close=AsyncMock())
    retired = SimpleNamespace(close=AsyncMock())
    registry._embeddings = embedding
    registry._retired_clients = [retired]
    container.infrastructure.ml.ml_clients = registry
    container.infrastructure.ml.file_storage = SimpleNamespace(aclose=AsyncMock())
    container.application.dispose = AsyncMock()
    container.infrastructure.events.dispose = AsyncMock()
    if failure == "application":
        container.application.dispose.side_effect = RuntimeError("cleanup failed")
    elif failure == "events":
        container.infrastructure.events.dispose.side_effect = RuntimeError("cleanup failed")
    elif failure == "storage":
        container.infrastructure.ml.file_storage.aclose.side_effect = RuntimeError("cleanup failed")
    if failure:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await container.dispose()
    else:
        await container.dispose()
        await container.dispose()
    embedding.close.assert_awaited_once()
    retired.close.assert_awaited_once()
    assert container.infrastructure.ml.ml_clients is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_worker_uses_container_teardown_before_external_connections(monkeypatch, failed):
    calls = []

    async def dispose():
        calls.append("container")
        if failed:
            raise RuntimeError("cleanup failed")

    async def close_redis():
        calls.append("redis")

    async def disconnect():
        calls.append("database")

    monkeypatch.setattr(redis_client, "aclose", close_redis)
    monkeypatch.setattr(database, "disconnect", disconnect)
    container = SimpleNamespace(dispose=dispose)
    listener = SimpleNamespace(stop=AsyncMock())
    ctx = {"container": container, "config_listener": listener}
    if failed:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await on_shutdown(ctx)
    else:
        await on_shutdown(ctx)
    assert calls == ["container", "redis", "database"]
    listener.stop.assert_not_awaited()
