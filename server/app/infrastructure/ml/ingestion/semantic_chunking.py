"""Semantic chunking — embedding-based text splitting.

Uses cosine similarity between adjacent sentence embeddings to find
natural topic boundaries. Falls back to sentence-level splitting when
embedding is unavailable.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("detailed")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])\n")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_break_points(
        similarities: list[float], threshold: float = 0.5
) -> list[int]:
    """Find indices where similarity drops below threshold (topic boundaries)."""
    breaks = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breaks.append(i + 1)  # break AFTER this sentence
    return breaks


def _fallback_sentence_split(text: str, max_chunk_size: int) -> list[str]:
    """Fallback: split by sentences, merge until max_chunk_size."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for sentence in sentences:
        if current_size + len(sentence) + 1 > max_chunk_size and current:
            chunks.append(" ".join(current))
            current = []
            current_size = 0
        current.append(sentence)
        current_size += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]
