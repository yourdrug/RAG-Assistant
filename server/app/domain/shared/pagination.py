"""Generic cursor-based pagination container."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from domain.shared.exceptions import ValidationError


@dataclass(frozen=True)
class Pagination:
    limit: int = 50
    offset: int = 0

    def __post_init__(self) -> None:
        if self.limit < 1 or self.offset < 0:
            message = "Pagination requires a positive limit and nonnegative offset"
            raise ValidationError(message)


@dataclass(frozen=True)
class CursorPage[T]:
    """A page of results returned by cursor-based (keyset) pagination.

    ``items`` holds the page rows in natural ascending order.
    ``next_cursor`` / ``prev_cursor`` are opaque base64url strings;
    *None* signals that there are no more rows in that direction.
    """

    items: list[T]
    next_cursor: str | None
    prev_cursor: str | None


def encode_cursor(chunk_index: int, chunk_id: int) -> str:
    """Encode a (chunk_index, chunk_id) pair into an opaque base64url cursor."""
    payload = f"{chunk_index}:{chunk_id}".encode()
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[int, int]:
    """Decode an opaque cursor back to ``(chunk_index, chunk_id)``.

    Raises ``ValidationError`` if the cursor is malformed.
    """
    try:
        # Re-pad to a multiple of 4 for urlsafe_b64decode
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded).decode("utf-8")
        return parse_cursor_parts(raw)
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        msg = f"Invalid cursor: {exc}"
        raise ValidationError(msg) from exc


def parse_cursor_parts(raw: str) -> tuple[int, int]:
    parts = raw.split(":")
    pair_length = 2

    if len(parts) != pair_length:
        msg = "expected exactly two parts"
        raise ValueError(msg)

    return int(parts[0]), int(parts[1])
