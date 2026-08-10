"""Postgres LISTEN/NOTIFY broadcaster for config change events.

Sends ``pg_notify('config_changed', payload)`` to alert other processes
about config parameter changes.  Uses a dedicated write session from the
database manager, or an existing session for transactional atomicity
when the config change originates from the same transaction.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from domain.events.config_events import ConfigParameterChanged
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from infrastructure.database.database import DatabaseManager

log = logging.getLogger("default")

_CHANNEL = "config_changed"
_MAX_PAYLOAD_BYTES = 7500


def _build_payload(event: ConfigParameterChanged) -> str:
    payload = json.dumps(
        {
            "key": event.key,
            "old_value": event.old_value,
            "new_value": event.new_value,
            "value_type": event.value_type,
            "changed_by": event.changed_by,
        }
    )
    if len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        payload = json.dumps({"key": event.key, "refetch": True})
    return payload


class PostgresConfigBroadcaster:
    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    async def broadcast(self, event: ConfigParameterChanged) -> None:
        session = self._database.get_write_session()
        try:
            await self._send_notify(session, event)
            await session.commit()
        except Exception:
            log.exception("Failed to send pg_notify for config_changed")
        finally:
            await session.close()

    async def broadcast_within_session(self, session: object, event: ConfigParameterChanged) -> None:
        """Send pg_notify within an existing session (same transaction as UPDATE)."""
        if not isinstance(session, AsyncSession):
            raise TypeError(f"Expected AsyncSession, got {type(session)}")
        try:
            await self._send_notify(session, event)
        except Exception:
            log.exception("Failed to send pg_notify within session for config_changed")

    @staticmethod
    async def _send_notify(session: AsyncSession, event: ConfigParameterChanged) -> None:
        payload = _build_payload(event)
        await session.execute(
            text("SELECT pg_notify(:channel, :payload)"),
            {"channel": _CHANNEL, "payload": payload},
        )
