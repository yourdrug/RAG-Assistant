"""Shared utilities for DI containers."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class ContainerNotInitializedError(RuntimeError):
    """Raised when accessing a dependency before the container is initialized."""


def require[T](value: T | None, name: str) -> T:
    """Raise ContainerNotInitializedError if *value* is None.

    Use for property accessors and factory methods that need to ensure
    the container has been initialized before returning a dependency.
    """
    if value is None:
        msg = f"{name} not initialized — Container.init() must be called first"
        raise ContainerNotInitializedError(msg)

    return value


def missing_fields(obj: DataclassInstance) -> list[str]:
    """Return names of all dataclass fields on *obj* that are ``None``."""
    return [f.name for f in dataclasses.fields(obj) if getattr(obj, f.name) is None]
