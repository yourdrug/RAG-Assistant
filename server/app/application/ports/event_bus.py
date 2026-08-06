"""EventBus port — абстракция шины доменных событий.

Application и Domain зависят только от этого протокола.
Конкретная реализация (in-process / Postgres LISTEN/NOTIFY / Kafka) — в infrastructure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class EventBus(Protocol):
    def publish(self, event: object) -> None: ...
    def subscribe(self, event_type: type, handler: Callable[[object], None]) -> None: ...
