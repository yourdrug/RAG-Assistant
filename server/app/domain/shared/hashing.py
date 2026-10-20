"""Shared hashing utilities."""

import hashlib


def content_hash(text: str) -> str:
    """Deterministic short hash for deduplication and merge keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
