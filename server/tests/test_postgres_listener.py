"""LISTEN transport uses manager-owned connections and releases its tasks."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from psycopg import Notify
import pytest

from infrastructure.events.postgres_listener import PostgresListener


def connection() -> MagicMock:
    async def notifications() -> AsyncIterator[Notify]:
        await asyncio.Event().wait()
        yield Notify("updates", "payload", 1)

    return MagicMock(closed=False, execute=AsyncMock(), close=AsyncMock(), notifies=notifications)


def database_manager(*connections: MagicMock | Exception) -> MagicMock:
    pending = iter(connections)

    @asynccontextmanager
    async def master_connection() -> AsyncIterator[MagicMock]:
        driver = next(pending)

        if isinstance(driver, Exception):
            raise driver

        raw = MagicMock(driver_connection=driver)
        try:
            yield MagicMock(get_raw_connection=AsyncMock(return_value=raw))
        finally:
            await driver.close()
            driver.closed = True

    return MagicMock(async_master_connection=MagicMock(side_effect=master_connection))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disconnect", "connect", "listen", "resync", "during_resync"])
async def test_reconnect_subscribes_then_resyncs(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    first, second = connection(), connection()
    connected = asyncio.Event()
    database = database_manager(RuntimeError() if failure == "connect" else first, second)
    monkeypatch.setattr("infrastructure.events.postgres_listener.RECONNECT_DELAY_SEC", 0)

    if failure == "listen":
        first.execute.side_effect = RuntimeError()

    async def disconnect() -> AsyncIterator[Notify]:
        yield Notify("updates", "pending", 1)
        raise ConnectionError

    first.notifies = disconnect
    notify = MagicMock()
    states: list[bool] = []

    async def resync() -> None:
        current = first if listener.connection is first else second
        assert current.execute.await_count == 1
        statement = current.execute.call_args.args[0]
        assert statement.as_string() == 'LISTEN "updates"'

        if current is first:
            if failure == "resync":
                raise RuntimeError

            if failure == "during_resync":
                first.closed = True
                await asyncio.sleep(0)
        else:
            connected.set()

    listener = PostgresListener(
        database, "updates", notify, resync, lambda: states.append(listener.is_connected)
    )
    await listener.start()
    try:
        await asyncio.wait_for(connected.wait(), 1)
        assert listener.connection is second
        assert database.async_master_connection.call_count == 2

        if failure != "connect":
            first.close.assert_awaited_once_with()
    finally:
        await listener.stop()
    second.close.assert_awaited_once_with()
    assert not listener.is_connected
    assert not listener.background_tasks
    assert listener.supervisor_task is None
    assert states[-1] is False
    assert True in states


@pytest.mark.asyncio
async def test_notifications_deliver_payload_and_release_generator() -> None:
    driver = connection()
    received = asyncio.Event()
    released = asyncio.Event()

    async def notifications() -> AsyncIterator[Notify]:
        try:
            yield Notify("updates", "buffered payload", 1)
            await asyncio.Event().wait()
        finally:
            released.set()

    driver.notifies = notifications
    notify = MagicMock(side_effect=lambda _payload: received.set())
    listener = PostgresListener(database_manager(driver), "updates", notify, AsyncMock())
    await listener.start()
    try:
        await asyncio.wait_for(received.wait(), 1)
        notify.assert_called_once_with("buffered payload")
    finally:
        await listener.stop()
    assert released.is_set()
    driver.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_stop_cancels_resync_and_notification_tasks() -> None:
    driver = connection()
    resync_started, notify_started = asyncio.Event(), asyncio.Event()
    cancelled: list[asyncio.Event] = []

    async def block(started: asyncio.Event) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(started)

    listener = PostgresListener(
        database_manager(driver), "updates", MagicMock(), lambda: block(resync_started)
    )
    await listener.start()
    task = listener.supervisor_task
    await listener.start()
    assert listener.supervisor_task is task
    await asyncio.wait_for(resync_started.wait(), 1)
    listener.schedule(block(notify_started))
    await asyncio.wait_for(notify_started.wait(), 1)
    await listener.stop()
    await listener.stop()
    assert cancelled == [resync_started, notify_started]
    driver.close.assert_awaited_once_with()
    assert not listener.background_tasks


@pytest.mark.asyncio
async def test_stop_during_reconnect_delay() -> None:
    database = database_manager(RuntimeError())
    failed = asyncio.Event()
    listener = PostgresListener(database, "updates", MagicMock(), AsyncMock(), failed.set)
    await listener.start()
    await asyncio.wait_for(failed.wait(), 1)
    await listener.stop()
    assert listener.supervisor_task is None
    assert not listener.is_connected
