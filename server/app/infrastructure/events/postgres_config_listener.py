"""Postgres LISTEN/NOTIFY listener for config_changed channel.

Uses a dedicated connection from DatabaseManager (outside the session pool) -- LISTEN
holds the connection open permanently, so it cannot be returned to the pool.
Includes a supervisor task for automatic reconnection on disconnect and a
full resync on (re)connect to close the window of lost NOTIFY messages.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from infrastructure.initialization import load_config_from_db
from infrastructure.events.postgres_listener import PostgresListener

from infrastructure.metrics.metrics import (
    CONFIG_LISTENER_CONNECTED,
    CONFIG_NOTIFY_RECEIVED_TOTAL,
    CONFIG_RESYNC_TOTAL,
)

if TYPE_CHECKING:
    from infrastructure.database.database import DatabaseManager

    from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
    from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")

_CHANNEL = "config_changed"


class PostgresConfigListener:
    """LISTEN config_changed с автопереподключением и resync.

    При каждом (пере)подключении выполняется полная resync из БД — это закрывает
    окно, в которое NOTIFY мог быть потерян во время offline-состояния.
    """

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        configuration: SettingsProviderConfiguration,
        database: DatabaseManager,
    ) -> None:
        self._uow_factory = uow_factory
        self.configuration = configuration
        self._sync_lock = asyncio.Lock()
        self.transport = PostgresListener(
            database, _CHANNEL, self.on_notify, self.on_connect, self.on_connection_changed
        )

    async def start(self) -> None:
        await self.transport.start()

    async def on_connect(self) -> None:
        await self.resync(trigger="reconnect")

    def on_connection_changed(self) -> None:
        CONFIG_LISTENER_CONNECTED.set(int(self.is_connected))

    async def resync(self, trigger: str = "manual") -> None:
        """Публичный метод для полной синхронизации config_parameters -> settings.

        Вызывается при (пере)подключении, периодическим scheduler'ом или вручную.
        """
        try:
            await self.synchronize()
            CONFIG_RESYNC_TOTAL.labels(trigger=trigger).inc()
        except Exception:
            self.configuration.failed()
            log.exception("Configuration snapshot resync failed")

    async def synchronize(self, *, require_complete=False) -> None:
        # Lock covers fetch + apply; an older concurrent fetch cannot apply last.
        async with self._sync_lock:
            await load_config_from_db(
                self._uow_factory,
                self.configuration,
                require_complete=require_complete,
            )

    def schedule_sync(self) -> None:
        self.transport.schedule(self.resync(trigger="notify"))

    def on_notify(self, payload: str) -> None:
        CONFIG_NOTIFY_RECEIVED_TOTAL.inc()
        self.schedule_sync()

    async def stop(self) -> None:
        await self.transport.stop()

    @property
    def is_connected(self) -> bool:
        return self.transport.is_connected
