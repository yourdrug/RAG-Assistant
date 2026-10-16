"""EventBus port -- abstraction for the domain event bus.

The application and domain layers depend only on this protocol.  Concrete
implementations (in-process, Postgres LISTEN/NOTIFY, Kafka) live in the
infrastructure layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class EventBus(Protocol):
    def publish(self, event: object) -> None: ...
    def subscribe(self, event_type: type, handler: Callable[[object], None]) -> None: ...
