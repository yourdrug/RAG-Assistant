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


# ---------------------------------------------------------------------------
# RRF merge — pure function, zero infrastructure dependencies
# ---------------------------------------------------------------------------


def rrf_merge(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[str]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Takes (content_hash, score) pairs from each source.
    Returns merged list of content hashes sorted by descending RRF score.
    k=60 is the standard constant from the original RRF paper.
    """
    rrf_scores: dict[str, float] = {}

    for rank, (h, _score) in enumerate(dense_results):
        rrf_scores[h] = rrf_scores.get(h, 0.0) + dense_weight / (k + rank + 1)

    for rank, (h, _score) in enumerate(sparse_results):
        rrf_scores[h] = rrf_scores.get(h, 0.0) + sparse_weight / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [h for h, _score in merged]


# ---------------------------------------------------------------------------
# Document deduplication — pure function
# ---------------------------------------------------------------------------


def deduplicate_docs(docs: list) -> list:
    """Remove near-duplicate chunks by content_hash to improve context diversity."""
    seen_hashes: set[str] = set()
    unique_docs: list = []
    for doc in docs:
        h = content_hash(doc.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)
    return unique_docs


# ---------------------------------------------------------------------------
# Percentile — pure function, zero dependencies
# ---------------------------------------------------------------------------


def percentile(sorted_data: list[float], p: float) -> float:
    """Compute percentile from pre-sorted data using linear interpolation."""
    if not sorted_data:
        return 0.0
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
