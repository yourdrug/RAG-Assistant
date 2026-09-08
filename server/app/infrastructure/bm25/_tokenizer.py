"""Text tokenization for BM25 indexing and search."""

from __future__ import annotations

import re

from infrastructure.bm25._stemmer import stem_token

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]{2,}", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, stem tokens. 2+ char tokens only."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [stem_token(t) for t in tokens]


def tokenize_raw(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric without stemming. For indexing."""
    return _TOKEN_RE.findall(text.lower())
