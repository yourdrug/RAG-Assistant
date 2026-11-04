"""Postgres LISTEN/NOTIFY broadcaster for config change events.

Sends ``pg_notify('config_changed', payload)`` to alert other processes
about config parameter changes.  Uses an existing session so the NOTIFY is
transactionally atomic with the config UPDATE that triggered it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


if TYPE_CHECKING:
    from application.runtime.config import ConfigParameterChanged


log = logging.getLogger("default")

_CHANNEL = "config_changed"


class PostgresConfigBroadcaster:
    async def broadcast_within_session(self, session: object, _event: ConfigParameterChanged) -> None:
        """Send pg_notify within an existing session (same transaction as UPDATE)."""
        if not isinstance(session, AsyncSession):
            msg = f"Expected AsyncSession, got {type(session)}"
            raise TypeError(msg)

        try:
            await session.execute(
                text("SELECT pg_notify(:channel, :payload)"),
                {"channel": _CHANNEL, "payload": ""},
            )
        except Exception:
            log.exception("Failed to send pg_notify within session for config_changed")
