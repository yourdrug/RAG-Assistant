"""Singleton decorator -- ensures only one instance of a class exists.

Usage::

    @Singleton
    class MyService:
        ...
"""

from __future__ import annotations

import threading
from typing import Any, TypeVar

T = TypeVar("T")


def Singleton(aClass: type[T]) -> T:
    """Turn a class into a singleton (thread-safe)."""
    _lock = threading.Lock()
    _instance: T | None = None

    class Wrapper:
        def __call__(self, *args: Any, **kwargs: Any) -> T:
            nonlocal _instance
            if _instance is None:
                with _lock:
                    if _instance is None:
                        _instance = aClass(*args, **kwargs)
            return _instance

    return Wrapper()  # type: ignore[return-value]
