"""Text tokenization for BM25 indexing and search."""

from __future__ import annotations

import re

from infrastructure.bm25.stemmer import stem_token

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, stem tokens; retain single digits but discard single letters."""
    return [stem_token(t) for t in tokenize_raw(text)]


def tokenize_raw(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric without stemming. For indexing."""
    return [t for t in _TOKEN_RE.findall(text.lower().replace("ё", "е")) if len(t) > 1 or t.isdigit()]
