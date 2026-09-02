"""Postgres LISTEN/NOTIFY broadcaster for config change events.

Sends ``pg_notify('config_changed', payload)`` to alert other processes
about config parameter changes.  Uses an existing session so the NOTIFY is
transactionally atomic with the config UPDATE that triggered it.
"""

from __future__ import annotations

import json
import logging

from domain.events.config_events import ConfigParameterChanged
from infrastructure.ml.config_subscribers import SENSITIVE_KEYS
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger("default")

_CHANNEL = "config_changed"
_MAX_PAYLOAD_BYTES = 7500


def _build_payload(event: ConfigParameterChanged) -> str:
    value_type = event.value_type
    if event.key in SENSITIVE_KEYS:
        payload = json.dumps(
            {
                "key": event.key,
                "value_type": value_type,
                "changed_by": event.changed_by,
                "refetch": True,
            }
        )
    else:
        payload = json.dumps(
            {
                "key": event.key,
                "old_value": event.old_value,
                "new_value": event.new_value,
                "value_type": value_type,
                "changed_by": event.changed_by,
            }
        )
    if len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        payload = json.dumps({"key": event.key, "refetch": True})
    return payload


class PostgresConfigBroadcaster:
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
