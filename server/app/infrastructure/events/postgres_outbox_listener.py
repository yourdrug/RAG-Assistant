"""Postgres LISTEN/NOTIFY listener for vector_outbox_ready channel.

Uses a dedicated raw asyncpg connection (not the SQLAlchemy pool) — LISTEN
holds the connection open permanently, so it cannot be returned to the pool.
Includes a supervisor task for automatic reconnection on disconnect and a
resync on (re)connect to close the window of lost NOTIFY messages.
"""

from __future__ import annotations

import asyncio
import json
import logging

import asyncpg

log = logging.getLogger("default")

_CHANNEL = "vector_outbox_ready"
_RECONNECT_DELAY_SEC = 5.0


class PostgresOutboxListener:
    """LISTEN/NOTIFY listener that triggers the outbox dispatcher on new entries."""

    def __init__(self, dispatcher, db_config: dict) -> None:
        """Initialize the outbox listener.

        Args:
            dispatcher: OutboxDispatcher instance to trigger on notifications.
            db_config: Dict with db_host, db_port, db_user, db_password, db_name.

        """
        self._dispatcher = dispatcher
        self._db_config = db_config
        self._conn: asyncpg.Connection | None = None
        self._stopped = False
        self._supervisor_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._stopped = False
        self._supervisor_task = asyncio.create_task(self._run_forever(), name="pg-outbox-listener")

    async def _run_forever(self) -> None:
        while not self._stopped:
            try:
                await self._connect_and_listen()
                disconnect_event = asyncio.Event()
                if self._conn is not None:
                    self._conn.add_termination_listener(lambda _c, _de=disconnect_event: _de.set())
                await disconnect_event.wait()
                if not self._stopped:
                    log.warning(
                        "Outbox LISTEN connection lost — reconnecting in %.0fs",
                        _RECONNECT_DELAY_SEC,
                    )
            except Exception:
                log.exception(
                    "Outbox LISTEN connection failed — retrying in %.0fs",
                    _RECONNECT_DELAY_SEC,
                )
            finally:
                await self._safe_close()

            if not self._stopped:
                await asyncio.sleep(_RECONNECT_DELAY_SEC)

    async def _connect_and_listen(self) -> None:
        self._conn = await asyncpg.connect(
            host=self._db_config["db_host"],
            port=int(self._db_config["db_port"]),
            user=self._db_config["db_user"],
            password=self._db_config["db_password"],
            database=self._db_config["db_name"],
        )
        await self._conn.add_listener(_CHANNEL, self._on_notify)
        log.info("Outbox LISTEN '%s' established", _CHANNEL)

        # Resync on connect — process any entries missed while disconnected
        await self.resync(trigger="reconnect")

    async def resync(self, trigger: str = "manual") -> None:
        """Process any pending outbox entries (safety net after reconnect)."""
        try:
            processed = await self._dispatcher.run_once(batch_size=100)
            if processed:
                log.info("Outbox resync (%s): processed %d entries", trigger, processed)
        except Exception:
            log.exception("Outbox resync failed")

    def _on_notify(self, connection: asyncpg.Connection, pid: int, channel: str, payload: str) -> None:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("Malformed outbox notification payload: %s", payload)
            return

        log.debug("Outbox NOTIFY received: %s", data)

        # Schedule dispatcher run as a background task — non-blocking
        task = asyncio.create_task(self._dispatcher.run_once())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _safe_close(self) -> None:
        if self._conn is not None and not self._conn.is_closed():
            try:
                await self._conn.close()
            except Exception:
                pass
        self._conn = None

    async def stop(self) -> None:
        self._stopped = True
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except asyncio.CancelledError:
                pass
        if self._background_tasks:
            for task in self._background_tasks:
                task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        await self._safe_close()
        log.info("Outbox LISTEN '%s' stopped", _CHANNEL)

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and not self._conn.is_closed()
