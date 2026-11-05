"""Regressions for ML invalidation and resource ownership."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from application.runtime.config import ConfigParameter, ConfigValueType
from composition.cli.commands.worker import on_shutdown
from composition.container import Container
from config import settings
from infrastructure.database.database import database
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from infrastructure.redis.redis_client import redis_client
from tests.fakes import require_present


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["llm_num_ctx_broad", "llm_num_predict_broad"])
async def test_broad_config_rebuilds_cached_llm(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    bus = InProcessEventBus()
    monkeypatch.setattr("infrastructure.events.in_process_event_bus.event_bus", bus)
    monkeypatch.setattr(settings, key, 8192)

    def create_client(breadth: object, *, runtime: object = None) -> SimpleNamespace:
        return SimpleNamespace(limit=getattr(settings, key), close=AsyncMock())

    monkeypatch.setattr("infrastructure.ml.clients.factories.create_llm_for_breadth", create_client)
    registry = MLClientRegistry()
    container = Container()
    monkeypatch.setattr(container.infrastructure.ml, "ml_clients", registry)

    monkeypatch.setattr("composition.infrastructure.event_bus", bus)
    monkeypatch.setattr(settings, "tei_embed_url", "http://tei-embed:8080")
    monkeypatch.setattr(settings, "tei_rerank_url", "http://tei-rerank:8080")
    container.infrastructure.configuration = SettingsProviderConfiguration(bus=bus)
    container.subscribe_config_events()
    try:
        cached = registry.llm_for_breadth("broad")
        assert cached.limit == 8192
        container.infrastructure.configuration.apply(
            [
                ConfigParameter(key, "8704", ConfigValueType.INT, "llm"),
            ]
        )
        current = registry.llm_for_breadth("broad")
        assert current is not cached
        assert current.limit == getattr(settings, key) == 8704
    finally:
        await container.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "application", "events", "storage"])
async def test_container_closes_ml_registry_even_if_other_cleanup_fails(
    failure: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = Container()
    registry = MLClientRegistry()
    embedding = SimpleNamespace(close=AsyncMock())
    retired = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(registry, "cached_embeddings", require_present(embedding))
    registry._retired_clients = [retired]
    monkeypatch.setattr(container.infrastructure.ml, "ml_clients", registry)
    monkeypatch.setattr(container.infrastructure.ml, "file_storage", SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr(container.application, "dispose", AsyncMock())
    monkeypatch.setattr(container.infrastructure.events, "dispose", AsyncMock())

    if failure == "application":
        cast("AsyncMock", container.application.dispose).side_effect = RuntimeError("cleanup failed")
    elif failure == "events":
        cast("AsyncMock", container.infrastructure.events.dispose).side_effect = RuntimeError(
            "cleanup failed"
        )
    elif failure == "storage":
        cast(
            "AsyncMock", require_present(require_present(container.infrastructure.ml.file_storage).aclose)
        ).side_effect = RuntimeError("cleanup failed")

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
async def test_worker_uses_container_teardown_before_external_connections(
    monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    calls = []

    async def dispose() -> None:
        calls.append("container")

        if failed:
            msg = "cleanup failed"
            raise RuntimeError(msg)

    async def close_redis() -> None:
        calls.append("redis")

    async def disconnect() -> None:
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
