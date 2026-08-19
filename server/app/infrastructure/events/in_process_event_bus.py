"""In-process synchronous EventBus implementation.

No external broker: handlers are registered in-process and called synchronously
on publish(). An exception in one subscriber must not crash others, so each
handler is wrapped in a try/except with logging.

Lifecycle:
  - Created at import time (module-level ``event_bus`` singleton)
  - Handlers subscribed in ``composition/events.py::_subscribe_config_events()``
  - Handlers unsubscribed in ``composition/events.py::_unsubscribe_config_events()``
    (called during ``Container.dispose()``)
  - Process-scoped: one instance per process, shared across all requests
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

    def unsubscribe_all(self) -> None:
        """Remove all registered handlers.

        Call during container dispose to avoid leaking handlers across
        container re-creation cycles (the event bus is a process-level singleton).
        """
        self._handlers.clear()
        log.debug("All event handlers removed")

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
