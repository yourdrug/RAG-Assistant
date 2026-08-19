"""Singleton decorator -- ensures only one instance of a class exists.

Usage::

    @Singleton
    class MyService:
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def Singleton(aClass: Any) -> Callable:
    """Turn a class into a singleton."""

    class Wrapper:
        instance: aClass = None

        def __call__(self, *args: tuple, **kwargs: dict) -> aClass:
            if self.instance is None:
                self.instance = aClass(*args, **kwargs)
            return self.instance

    return Wrapper()
