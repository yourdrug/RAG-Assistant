"""ConfigChangeBroadcaster port — abstraction for notifying other processes about config changes.

Application layer depends only on this protocol. Concrete implementation
(Postgres LISTEN/NOTIFY, Redis Pub/Sub, Kafka) lives in infrastructure.
"""

from __future__ import annotations

from typing import Protocol

from domain.events.config_events import ConfigParameterChanged

from application.ports.session_protocol import SessionProtocol


class ConfigChangeBroadcaster(Protocol):
    async def broadcast(self, event: ConfigParameterChanged) -> None: ...

    async def broadcast_within_session(
        self, session: SessionProtocol, event: ConfigParameterChanged
    ) -> None: ...
