"""Config notifications trigger a validated snapshot reload from the writer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from domain.runtime.config import ConfigParameterChanged
from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster
from infrastructure.events.postgres_config_listener import PostgresConfigListener


@pytest.fixture
def listener():
    return PostgresConfigListener(MagicMock(), configuration=MagicMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["", "not json", '{"key":"llm_model","new_value":"stale"}'])
async def test_notify_reloads_snapshot_regardless_of_payload(listener, monkeypatch, payload):
    synchronize = AsyncMock()
    monkeypatch.setattr(listener, "synchronize", synchronize)
    listener.on_notify(None, None, "config_changed", payload)
    await asyncio.gather(*listener._background_tasks)
    synchronize.assert_awaited_once_with()
    listener.configuration.failed.assert_not_called()


@pytest.mark.asyncio
async def test_resync_marks_configuration_unready_on_failure(listener, monkeypatch, caplog):
    monkeypatch.setattr(listener, "synchronize", AsyncMock(side_effect=RuntimeError("DB down")))
    await listener.resync(trigger="periodic")
    listener.configuration.failed.assert_called_once_with()
    assert "Configuration snapshot resync failed" in caplog.text


@pytest.mark.asyncio
async def test_connect_reloads_snapshot_after_listening(listener, monkeypatch):
    connection = AsyncMock()
    monkeypatch.setattr(
        "infrastructure.events.postgres_config_listener.asyncpg.connect",
        AsyncMock(return_value=connection),
    )
    synchronize = AsyncMock()

    async def synchronized_after_listen():
        connection.add_listener.assert_awaited_once_with("config_changed", listener.on_notify)
        await synchronize()

    monkeypatch.setattr(listener, "synchronize", synchronized_after_listen)
    await listener.connect_and_listen()
    synchronize.assert_awaited_once_with()
    assert listener._conn is connection


@pytest.mark.asyncio
async def test_stop_cancels_pending_sync_and_closes_connection(listener, monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def synchronize():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(listener, "synchronize", synchronize)
    listener._conn = MagicMock(is_closed=MagicMock(return_value=False), close=AsyncMock())
    connection = listener._conn
    listener.on_notify(None, None, "config_changed", "")
    await started.wait()
    await listener.stop()
    assert cancelled.is_set()
    assert not listener._background_tasks
    assert not listener.is_connected
    connection.close.assert_awaited_once_with()
    listener.configuration.failed.assert_not_called()


def test_is_connected_false_initially(listener):
    assert not listener.is_connected


@pytest.mark.asyncio
@pytest.mark.parametrize("key,value", [("llm_model", "x" * 9000), ("openrouter_api_key", "secret")])
async def test_broadcast_sends_only_invalidation_in_existing_transaction(key, value):
    session = AsyncMock(spec=AsyncSession)
    event = ConfigParameterChanged(key=key, old_value="old", new_value=value, value_type="str")
    await PostgresConfigBroadcaster().broadcast_within_session(session, event)
    statement, parameters = session.execute.call_args.args
    assert str(statement) == "SELECT pg_notify(:channel, :payload)"
    assert parameters == {"channel": "config_changed", "payload": ""}
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()
