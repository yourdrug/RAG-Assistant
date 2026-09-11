"""Postgres LISTEN/NOTIFY listener for config_changed channel.

Uses a dedicated raw asyncpg connection (not the SQLAlchemy pool) -- LISTEN
holds the connection open permanently, so it cannot be returned to the pool.
Includes a supervisor task for automatic reconnection on disconnect and a
full resync on (re)connect to close the window of lost NOTIFY messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import asyncpg
from application.ports.event_bus import EventBus
from config import settings
from domain.events.config_events import ConfigParameterChanged

from infrastructure.metrics.metrics import (
    CONFIG_LISTENER_CONNECTED,
    CONFIG_NOTIFY_RECEIVED_TOTAL,
    CONFIG_RESYNC_TOTAL,
)
from infrastructure.ml.config.config_subscribers import SENSITIVE_KEYS as _SENSITIVE_KEYS

if TYPE_CHECKING:
    from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
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
        event_bus: EventBus,
        uow_factory: UnitOfWorkFactory,
        domain_settings: DomainSettingsAdapter | None = None,
    ) -> None:
        self._bus = event_bus
        self._uow_factory = uow_factory
        self._domain_settings = domain_settings
        self._conn: asyncpg.Connection | None = None
        self._stopped = False
        self._supervisor_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._stopped = False
        self._supervisor_task = asyncio.create_task(self._run_forever(), name="pg-config-listener")

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
                        "Postgres LISTEN connection lost — reconnecting in %.0fs", _RECONNECT_DELAY_SEC
                    )
            except Exception:
                log.exception("Postgres LISTEN connection failed — retrying in %.0fs", _RECONNECT_DELAY_SEC)
            finally:
                await self._safe_close()

            if not self._stopped:
                await asyncio.sleep(_RECONNECT_DELAY_SEC)

    async def _connect_and_listen(self) -> None:
        self._conn = await asyncpg.connect(
            host=settings.db_host,
            port=int(settings.db_port),
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
        )
        await self._conn.add_listener(_CHANNEL, self._on_notify)
        CONFIG_LISTENER_CONNECTED.set(1)
        log.info("Postgres LISTEN '%s' established", _CHANNEL)

        await self.resync(trigger="reconnect")

    async def resync(self, trigger: str = "manual") -> None:
        """Публичный метод для полной синхронизации config_parameters -> settings.

        Вызывается при (пере)подключении, периодическим scheduler'ом или вручную.
        """
        await self._resync_all(trigger=trigger)

    async def _resync_all(self, trigger: str = "manual") -> None:
        try:
            async with self._uow_factory.create() as uow:
                rows = await uow.config_parameters.get_all()
            applied = 0
            for r in rows:
                if getattr(r, "domain_key", None) is not None:
                    # Check if the cached domain value actually differs from DB.
                    # Without this comparison resync fires spurious events on every
                    # cycle (old_value=None, same new_value) flooding audit logs.
                    domain_key: str = r.domain_key  # type: ignore[assignment]
                    if self._domain_settings is not None:
                        try:
                            cached = self._domain_settings.get(r.key, domain_key)
                        except KeyError:
                            cached = None
                        if cached is not None and self._values_equal(cached, r.value):
                            continue
                    self._bus.publish(
                        ConfigParameterChanged(
                            key=r.key,
                            old_value=cached if self._domain_settings is not None else None,
                            new_value=r.value,
                            value_type=r.value_type,
                            domain_key=r.domain_key,
                        )
                    )
                    applied += 1
                    continue
                current = getattr(settings, r.key, None)
                if self._values_equal(current, r.value):
                    continue
                # SENSITIVE_KEYS are never hot-reloaded (apply_to_settings skips
                # them).  If the in-memory value exists, the DB row is stale —
                # don't fire a spurious change event on every resync cycle.
                if r.key in _SENSITIVE_KEYS and current is not None:
                    continue
                self._bus.publish(
                    ConfigParameterChanged(
                        key=r.key,
                        old_value=json.dumps(current) if isinstance(current, list | dict) else str(current),
                        new_value=r.value,
                        value_type=r.value_type,
                        domain_key=r.domain_key,
                    )
                )
                applied += 1
            CONFIG_RESYNC_TOTAL.labels(trigger=trigger).inc()
            log.info("Config resync (%s): %d/%d parameters changed", trigger, applied, len(rows))
        except Exception:
            log.exception("Config resync failed — settings may be stale until next successful resync")

    def _on_notify(self, connection: asyncpg.Connection, pid: int, channel: str, payload: str) -> None:
        CONFIG_NOTIFY_RECEIVED_TOTAL.inc()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log.exception("Malformed config_changed payload: %s", payload)
            return

        if data.get("refetch"):
            task = asyncio.create_task(self._refetch_and_publish(data["key"], data.get("domain_key")))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return

        event = ConfigParameterChanged(
            key=data["key"],
            old_value=data.get("old_value"),
            new_value=data["new_value"],
            value_type=data["value_type"],
            changed_by=data.get("changed_by"),
            domain_key=data.get("domain_key"),
        )
        self._bus.publish(event)

    async def _refetch_and_publish(self, key: str, domain_key: str | None = None) -> None:
        """Fallback для payload'ов, не влезших в лимит pg_notify (8000 байт)."""
        try:
            async with self._uow_factory.create() as uow:
                param = await uow.config_parameters.get_by_key_and_domain(key, domain_key)
            if param is not None:
                self._bus.publish(
                    ConfigParameterChanged(
                        key=param.key,
                        old_value=None,
                        new_value=param.value,
                        value_type=param.value_type,
                        domain_key=param.domain_key,
                    )
                )
        except Exception:
            log.exception("Failed to refetch config parameter '%s' after refetch notification", key)

    async def _safe_close(self) -> None:
        CONFIG_LISTENER_CONNECTED.set(0)
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
        log.info("Postgres LISTEN '%s' stopped", _CHANNEL)

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and not self._conn.is_closed()

    @staticmethod
    def _normalize_to_comparable(current: object) -> object:
        """Normalize a Python object to its JSON-equivalent form for comparison."""
        if isinstance(current, bool):
            return json.loads(json.dumps(current))
        if isinstance(current, list | dict | int | float):
            return json.loads(json.dumps(current, ensure_ascii=False, sort_keys=True))
        return json.loads(json.dumps(str(current), ensure_ascii=False))

    @staticmethod
    def _normalize_types(a: object, b: object) -> tuple[object, object]:
        """Coerce str<->numeric mismatches so mixed types compare correctly."""
        if isinstance(a, str) and isinstance(b, (int, float)):
            try:
                return float(a), b
            except (ValueError, TypeError):
                return a, b
        if isinstance(b, str) and isinstance(a, (int, float)):
            try:
                return a, float(b)
            except (ValueError, TypeError):
                return a, b
        return a, b

    @staticmethod
    def _values_equal(current: object, db_value: str | None) -> bool:
        """Compare in-memory setting with DB-stored JSON value, ignoring format differences.

        DB stores JSON (``["ru","en"]``), while Python objects serialise as repr
        (``['ru', 'en']``).  We parse both sides to JSON and compare tokens so
        that whitespace / quote style doesn't trigger a false "changed" event.
        """
        if current is None and db_value is None:
            return True
        if current is None or db_value is None:
            return False
        try:
            db_parsed = json.loads(db_value)
        except (json.JSONDecodeError, TypeError):
            return str(current) == db_value
        current_parsed = PostgresConfigListener._normalize_to_comparable(current)
        current_normalized, db_normalized = PostgresConfigListener._normalize_types(current_parsed, db_parsed)
        return current_normalized == db_normalized
