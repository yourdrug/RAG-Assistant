"""Thread-safe singleton decorator."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


def singleton[**P, T](factory: Callable[P, T]) -> Callable[P, T]:
    """Create the wrapped instance once, including across concurrent callers."""
    lock = threading.Lock()
    instance: T | None = None

    def get_instance(*args: P.args, **kwargs: P.kwargs) -> T:
        nonlocal instance

        if instance is None:
            with lock:
                if instance is None:
                    instance = factory(*args, **kwargs)

        return instance

    return get_instance
