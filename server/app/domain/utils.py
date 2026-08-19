"""Domain utilities -- pure functions with no infrastructure dependencies."""

from __future__ import annotations

import hashlib


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
