"""Dedicated LISTEN connection with reconnect and owned background tasks."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress
import logging
from typing import TYPE_CHECKING, Any, cast

from psycopg import sql


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from psycopg import AsyncConnection

    from infrastructure.database.database import DatabaseManager

log = logging.getLogger("default")
RECONNECT_DELAY_SEC = 5.0


class PostgresListener:
    """Own the transport; channel-specific behavior is supplied as callbacks."""

    def __init__(
        self,
        database: DatabaseManager,
        channel: str,
        on_notify: Callable[[str], None],
        on_connect: Callable[[], Coroutine[Any, Any, None]],
        on_connection_changed: Callable[[], None] | None = None,
    ) -> None:
        self.database = database
        self.channel = channel
        self.on_notify = on_notify
        self.on_connect = on_connect
        self.on_connection_changed = on_connection_changed
        self.connection: AsyncConnection[Any] | None = None
        self.supervisor_task: asyncio.Task[None] | None = None
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.stopped = False

    async def start(self) -> None:
        if self.supervisor_task is None or self.supervisor_task.done():
            self.stopped = False
            self.supervisor_task = asyncio.create_task(self.run_forever(), name=f"pg-listener-{self.channel}")

    async def run_forever(self) -> None:
        while not self.stopped:
            try:
                # The manager owns a dedicated NullPool connection in AUTOCOMMIT mode.
                async with self.database.async_master_connection() as connection:
                    raw = await connection.get_raw_connection()
                    self.connection = cast("AsyncConnection[Any]", raw.driver_connection)
                    await self.connection.execute(sql.SQL("LISTEN {}").format(sql.Identifier(self.channel)))
                    self.connection_changed()
                    log.info("Postgres LISTEN '%s' established", self.channel)
                    await self.on_connect()
                    # psycopg buffers notifications received between LISTEN and resync.
                    async with aclosing(self.connection.notifies()) as notifications:
                        async for notification in notifications:
                            self.on_notify(notification.payload)
            except Exception:
                log.exception(
                    "Postgres LISTEN '%s' failed — retrying in %.0fs", self.channel, RECONNECT_DELAY_SEC
                )
            finally:
                self.connection = None
                self.connection_changed()

            if not self.stopped:
                await asyncio.sleep(RECONNECT_DELAY_SEC)

    def schedule(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coroutine)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    def connection_changed(self) -> None:
        if self.on_connection_changed is not None:
            self.on_connection_changed()

    async def stop(self) -> None:
        self.stopped = True

        if self.supervisor_task is not None:
            self.supervisor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.supervisor_task
            self.supervisor_task = None

        tasks = tuple(self.background_tasks)

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self.background_tasks.clear()
        log.info("Postgres LISTEN '%s' stopped", self.channel)

    @property
    def is_connected(self) -> bool:
        return self.connection is not None and not self.connection.closed
