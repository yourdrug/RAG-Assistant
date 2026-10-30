"""Rank fusion, provision diversity and content deduplication."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from domain.knowledge.chunk import SENTENCE_UNIT_KIND
from domain.shared.hashing import content_hash


if TYPE_CHECKING:
    from collections.abc import Sequence


class RetrievalDocument(Protocol):
    """Read-only content and metadata required by retrieval policies."""

    @property
    def page_content(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


def prioritize_distinct_provisions[ItemT: RetrievalDocument | tuple[RetrievalDocument, float]](
    items: list[ItemT],
) -> list[ItemT]:
    """Stable first pass over provisions, followed by their remaining chunks.

    Accept documents or (document, score) pairs. Sentence splits share their
    parent's provision. Unstructured chunks and tables retain their own slots.
    Scope includes the document/version so unrelated provisions never collapse.
    No text is removed: remaining chunks can fill any unused budget.
    """
    seen = set()
    first: list[ItemT] = []
    remaining: list[ItemT] = []

    for item in items:
        doc = item[0] if isinstance(item, tuple) else item
        metadata = doc.metadata
        section = metadata.get("section") or metadata.get("heading")
        source = metadata.get("document_id") or metadata.get("source")

        if not section or not source or metadata.get("table_id"):
            first.append(item)
            continue

        parts = section.replace(" \u203a ", " > ").split(" > ")

        if parts[-1] == SENTENCE_UNIT_KIND:
            parts = parts[:-1]

        if not parts:
            first.append(item)
            continue

        key = (source, metadata.get("act_version_id"), tuple(parts))

        if key in seen:
            remaining.append(item)
        else:
            seen.add(key)
            first.append(item)

    return first + remaining


def compute_reranker_score(sources: Sequence[object]) -> float | None:
    """Extract the maximum reranker score from a list of source dicts."""
    if not sources:
        return None

    scores = [s.get("max_score", 0) for s in sources if isinstance(s, dict)]
    return max(scores) if scores else None


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


def deduplicate_docs[DocumentT: RetrievalDocument](docs: list[DocumentT]) -> list[DocumentT]:
    """Remove near-duplicate chunks by content_hash to improve context diversity."""
    seen_hashes: set[str] = set()
    unique_docs: list[DocumentT] = []

    for doc in docs:
        h = content_hash(doc.page_content)

        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    return unique_docs
