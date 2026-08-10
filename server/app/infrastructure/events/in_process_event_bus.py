"""In-process synchronous EventBus implementation.

No external broker: handlers are registered in-process and called synchronously
on publish(). An exception in one subscriber must not crash others, so each
handler is wrapped in a try/except with logging.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable

log = logging.getLogger("default")


class InProcessEventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable) -> None:
        self._handlers[event_type].append(handler)
        log.debug(
            "Subscribed %s -> %s",
            event_type.__name__,
            getattr(handler, "__qualname__", handler),
        )

    def publish(self, event: object) -> None:
        for handler in self._handlers.get(type(event), []):
            try:
                handler(event)
            except Exception:
                log.exception(
                    "EventBus handler %s failed for %s",
                    getattr(handler, "__qualname__", handler),
                    type(event).__name__,
                )


event_bus = InProcessEventBus()
