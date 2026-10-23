"""Outbox-specific callbacks retain notification and reconnect behavior."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from infrastructure.events.postgres_outbox_listener import PostgresOutboxListener


@pytest.mark.asyncio
async def test_reconnect_processes_missed_entries() -> None:
    dispatcher = MagicMock(run_once=AsyncMock(return_value=2))
    listener = PostgresOutboxListener(dispatcher, MagicMock())
    await listener.on_connect()
    dispatcher.run_once.assert_awaited_once_with(batch_size=100)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ['{"id": 1}', "not json"])
async def test_only_valid_notifications_schedule_dispatch(payload: str) -> None:
    dispatcher = MagicMock(run_once=AsyncMock())
    listener = PostgresOutboxListener(dispatcher, MagicMock())
    listener.on_notify(payload)
    await asyncio.gather(*listener.transport.background_tasks)

    if payload == "not json":
        dispatcher.run_once.assert_not_awaited()
    else:
        dispatcher.run_once.assert_awaited_once_with()

    await listener.stop()


@pytest.mark.asyncio
async def test_resync_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    listener = PostgresOutboxListener(
        MagicMock(run_once=AsyncMock(side_effect=RuntimeError("offline"))), MagicMock()
    )
    await listener.resync()
    assert "Outbox resync failed" in caplog.text
