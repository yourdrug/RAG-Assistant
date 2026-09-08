"""Generic cursor-based pagination container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CursorPage(Generic[T]):
    """A page of results returned by cursor-based (keyset) pagination.

    ``items`` holds the page rows in natural ascending order.
    ``next_cursor`` / ``prev_cursor`` are opaque base64url strings;
    *None* signals that there are no more rows in that direction.
    """

    items: list[T]
    next_cursor: str | None
    prev_cursor: str | None
