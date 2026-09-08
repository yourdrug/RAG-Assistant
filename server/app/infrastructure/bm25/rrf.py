"""Reciprocal Rank Fusion (RRF) — pure function, zero dependencies."""

from __future__ import annotations


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
