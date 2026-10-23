"""Real manager-owned LISTEN connections buffer notifications and reconnect."""

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from infrastructure.database.database import DatabaseManager
from infrastructure.events.postgres_listener import PostgresListener


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_listener_buffers_resync_notifications_and_reconnects(  # noqa: PLR0915
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = DatabaseManager()
    database.create_master_connection_engines(pg_engine.url.set(drivername="postgresql+psycopg"), {})
    channel = f"test_listener_{uuid4().hex}"
    received = asyncio.Event()
    payloads: list[str] = []
    reconnects = 0
    monkeypatch.setattr("infrastructure.events.postgres_listener.RECONNECT_DELAY_SEC", 0)

    def on_notify(payload: str) -> None:
        payloads.append(payload)
        received.set()

    async def on_connect() -> None:
        nonlocal reconnects
        reconnects += 1
        # Deliver before the notification iterator starts, while resync is running.
        async with database.async_master_connection() as connection:
            await connection.execute(
                text("SELECT pg_notify(:channel, :payload)"), {"channel": channel, "payload": str(reconnects)}
            )

    listener = PostgresListener(database, channel, on_notify, on_connect)
    await listener.start()
    try:
        await asyncio.wait_for(received.wait(), 5)
        assert payloads == ["1"]
        first = listener.connection
        assert first is not None
        received.clear()
        async with database.async_master_connection() as connection:
            await connection.execute(
                text("SELECT pg_terminate_backend(:pid)"), {"pid": first.info.backend_pid}
            )
        await asyncio.wait_for(received.wait(), 5)
        assert payloads == ["1", "2"]
        assert first.closed
        assert listener.is_connected
    finally:
        await listener.stop()
        await database.disconnect()
    assert listener.supervisor_task is None
    assert not listener.background_tasks
