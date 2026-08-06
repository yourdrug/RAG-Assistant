"""In-process синхронная реализация EventBus.

Без внешнего брокера: обработчики регистрируются в памяти процесса и вызываются
синхронно при publish(). Исключение в одном подписчике не должно ронять остальных —
поэтому каждый handler оборачивается в try/except с логированием.
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
