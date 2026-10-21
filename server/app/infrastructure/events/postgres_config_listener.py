"""Postgres LISTEN/NOTIFY listener for config_changed channel.

Uses a dedicated raw asyncpg connection (not the SQLAlchemy pool) -- LISTEN
holds the connection open permanently, so it cannot be returned to the pool.
Includes a supervisor task for automatic reconnection on disconnect and a
full resync on (re)connect to close the window of lost NOTIFY messages.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import asyncpg
from config import settings
from infrastructure.initialization import load_config_from_db

from infrastructure.metrics.metrics import (
    CONFIG_LISTENER_CONNECTED,
    CONFIG_NOTIFY_RECEIVED_TOTAL,
    CONFIG_RESYNC_TOTAL,
)

if TYPE_CHECKING:
    from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
    from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")

_CHANNEL = "config_changed"
_RECONNECT_DELAY_SEC = 5.0


class PostgresConfigListener:
    """LISTEN config_changed с автопереподключением и resync.

    add_listener сам по себе не сообщает о разрыве соединения — используем
    connection.add_termination_listener, чтобы узнать о разрыве, и отдельный
    supervisor-таск, который переустанавливает соединение с задержкой.

    При каждом (пере)подключении выполняется полная resync из БД — это закрывает
    окно, в которое NOTIFY мог быть потерян во время offline-состояния.
    """

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        configuration: SettingsProviderConfiguration,
    ) -> None:
        self._uow_factory = uow_factory
        self.configuration = configuration
        self._sync_lock = asyncio.Lock()
        self._conn: asyncpg.Connection | None = None
        self._stopped = False
        self._supervisor_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._stopped = False
        self._supervisor_task = asyncio.create_task(self.run_forever(), name="pg-config-listener")

    async def run_forever(self) -> None:
        while not self._stopped:
            try:
                await self.connect_and_listen()
                disconnect_event = asyncio.Event()
                if self._conn is not None:
                    self._conn.add_termination_listener(lambda _c, _de=disconnect_event: _de.set())
                await disconnect_event.wait()
                if not self._stopped:
                    log.warning(
                        "Postgres LISTEN connection lost — reconnecting in %.0fs", _RECONNECT_DELAY_SEC
                    )
            except Exception:
                log.exception("Postgres LISTEN connection failed — retrying in %.0fs", _RECONNECT_DELAY_SEC)
            finally:
                await self.safe_close()

            if not self._stopped:
                await asyncio.sleep(_RECONNECT_DELAY_SEC)

    async def connect_and_listen(self) -> None:
        self._conn = await asyncpg.connect(
            host=settings.db_host,
            port=int(settings.db_port),
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
        )
        await self._conn.add_listener(_CHANNEL, self.on_notify)
        CONFIG_LISTENER_CONNECTED.set(1)
        log.info("Postgres LISTEN '%s' established", _CHANNEL)

        await self.resync(trigger="reconnect")

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
        task = asyncio.create_task(self.resync(trigger="notify"))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def on_notify(self, connection: asyncpg.Connection, pid: int, channel: str, payload: str) -> None:
        CONFIG_NOTIFY_RECEIVED_TOTAL.inc()
        self.schedule_sync()

    async def safe_close(self) -> None:
        CONFIG_LISTENER_CONNECTED.set(0)
        if self._conn is not None and not self._conn.is_closed():
            try:
                await self._conn.close()
            except Exception:
                log.debug("Failed to close config listener connection", exc_info=True)
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
        await self.safe_close()
        log.info("Postgres LISTEN '%s' stopped", _CHANNEL)

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and not self._conn.is_closed()
