"""Domain utilities -- pure functions with no infrastructure dependencies."""

from __future__ import annotations

import base64
import hashlib

from domain.exceptions import ValidationError


def content_hash(text: str) -> str:
    """Deterministic short hash for deduplication and merge keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_bool(value: str) -> bool:
    """Parse a string to boolean, raising ValueError on failure."""
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"Cannot parse '{value}' as boolean")


def compute_reranker_score(sources: list[dict]) -> float | None:
    """Extract the maximum reranker score from a list of source dicts."""
    if not sources:
        return None
    scores = [s.get("max_score", 0) for s in sources if isinstance(s, dict)]
    return max(scores) if scores else None


# ---------------------------------------------------------------------------
# Cursor encoding for keyset pagination
# ---------------------------------------------------------------------------


def encode_cursor(chunk_index: int, chunk_id: int) -> str:
    """Encode a (chunk_index, chunk_id) pair into an opaque base64url cursor."""
    payload = f"{chunk_index}:{chunk_id}".encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[int, int]:
    """Decode an opaque cursor back to ``(chunk_index, chunk_id)``.

    Raises ``ValidationError`` if the cursor is malformed.
    """
    try:
        # Re-pad to a multiple of 4 for urlsafe_b64decode
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded).decode("utf-8")
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError("expected exactly two parts")
        return int(parts[0]), int(parts[1])
    except Exception as exc:
        raise ValidationError(f"Invalid cursor: {exc}") from exc
