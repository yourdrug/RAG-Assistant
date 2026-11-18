"""Shutdown steps keep running when an earlier step fails."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from composition import infrastructure as infrastructure_module
from composition.cli.commands import ingest
from composition.infrastructure import EventContainer, InfrastructureContainer
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client


@pytest.mark.asyncio
async def test_judge_clients_closed_when_corpus_close_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    close_judges = AsyncMock()
    monkeypatch.setattr(infrastructure_module, "close_judge_clients", close_judges)
    infra = InfrastructureContainer()
    infra.corpus_control = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("db down")))  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="db down"):
        await infra.close_corpus_and_judges()

    close_judges.assert_awaited_once()


@pytest.mark.asyncio
async def test_outbox_listener_stopped_when_config_listener_stop_fails() -> None:
    outbox_listener = SimpleNamespace(stop=AsyncMock())
    events = EventContainer(
        config_listener=SimpleNamespace(stop=AsyncMock(side_effect=RuntimeError("stop failed"))),  # type: ignore[arg-type]
        outbox_listener=outbox_listener,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        await events.dispose()

    outbox_listener.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cli_disconnects_partially_connected_database(monkeypatch: pytest.MonkeyPatch) -> None:
    infra = SimpleNamespace(init=lambda _database: None, dispose=AsyncMock())
    disconnect = AsyncMock()
    monkeypatch.setattr(ingest, "InfrastructureContainer", lambda: infra)
    monkeypatch.setattr(database, "connect", AsyncMock(side_effect=RuntimeError("replica down")))
    monkeypatch.setattr(database, "disconnect", disconnect)
    monkeypatch.setattr(redis_client, "init", AsyncMock())
    monkeypatch.setattr(redis_client, "aclose", AsyncMock())
    action = AsyncMock()

    with pytest.raises(RuntimeError, match="replica down"):
        await ingest.run_with_infrastructure(action)

    action.assert_not_awaited()
    disconnect.assert_awaited_once()
