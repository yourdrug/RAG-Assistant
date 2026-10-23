"""Postgres LISTEN/NOTIFY listener for vector_outbox_ready channel.

Uses a dedicated connection from DatabaseManager (outside the session pool) — LISTEN
holds the connection open permanently, so it cannot be returned to the pool.
Includes a supervisor task for automatic reconnection on disconnect and a
resync on (re)connect to close the window of lost NOTIFY messages.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING


from infrastructure.events.postgres_listener import PostgresListener

if TYPE_CHECKING:
    from infrastructure.database.database import DatabaseManager

log = logging.getLogger("default")

_CHANNEL = "vector_outbox_ready"


class PostgresOutboxListener:
    """LISTEN/NOTIFY listener that triggers the outbox dispatcher on new entries."""

    def __init__(self, dispatcher, database: DatabaseManager) -> None:
        """Initialize the outbox listener.

        Args:
            dispatcher: OutboxDispatcher instance to trigger on notifications.
            database: Manager providing dedicated master connections.

        """
        self._dispatcher = dispatcher
        self.transport = PostgresListener(database, _CHANNEL, self.on_notify, self.on_connect)

    async def start(self) -> None:
        await self.transport.start()

    async def on_connect(self) -> None:
        await self.resync(trigger="reconnect")

    async def resync(self, trigger: str = "manual") -> None:
        """Process any pending outbox entries (safety net after reconnect)."""
        try:
            processed = await self._dispatcher.run_once(batch_size=100)

            if processed:
                log.info("Outbox resync (%s): processed %d entries", trigger, processed)
        except Exception:
            log.exception("Outbox resync failed")

    def on_notify(self, payload: str) -> None:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("Malformed outbox notification payload: %s", payload)
            return

        log.debug("Outbox NOTIFY received: %s", data)

        # Schedule dispatcher run as a background task — non-blocking
        self.transport.schedule(self._dispatcher.run_once())

    async def stop(self) -> None:
        await self.transport.stop()

    @property
    def is_connected(self) -> bool:
        return self.transport.is_connected
