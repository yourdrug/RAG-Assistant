"""Rank fusion, provision diversity and content deduplication."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

from domain.knowledge.chunk import CHUNK_CONTEXT_FIELDS, SENTENCE_UNIT_KIND
from domain.knowledge.enums import PageContentType
from domain.rag.act_references import referenced_act
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
    """Prefer distinct provisions without trading away strong continuations.

    Unscored retrieval keeps the diversity pass. Scored evidence can retain up
    to three chunks of a provision ahead of a substantially weaker provision.
    Nearby scores still favor diversity; source and version boundaries are kept.
    """
    seen = set()
    first: list[ItemT] = []
    remaining: list[ItemT] = []

    for item in items:
        doc = item[0] if isinstance(item, tuple) else item
        key = provision_identity(doc)

        if key is None or key not in seen:
            first.append(item)

            if key is not None:
                seen.add(key)
        else:
            remaining.append(item)

    return rank_provision_continuations(first, remaining)


def rank_provision_continuations[ItemT: RetrievalDocument | tuple[RetrievalDocument, float]](
    first: list[ItemT],
    remaining: list[ItemT],
) -> list[ItemT]:
    """Allow bounded strong continuations ahead of weaker distinct evidence."""
    ranked: list[ItemT] = []
    anchors: dict[tuple[object, ...], RetrievalDocument] = {}
    counts: dict[tuple[object, ...], int] = {}

    for item in first:
        if isinstance(item, tuple):
            for continuation in list(remaining):
                if not isinstance(continuation, tuple):
                    continue

                doc, score = continuation
                key = provision_identity(doc)

                if (
                    key in anchors
                    and counts[key] < MAX_PROVISION_CHUNKS
                    and score * CONTINUATION_SCORE_RATIO > item[1]
                    and same_evidence_block(anchors[key], doc)
                ):
                    ranked.append(continuation)
                    remaining.remove(continuation)
                    counts[key] += 1

        ranked.append(item)
        doc = item[0] if isinstance(item, tuple) else item
        key = provision_identity(doc)

        if key is not None:
            anchors[key] = doc
            counts[key] = 1

    return ranked + remaining


MAX_PROVISION_CHUNKS = 3
CONTINUATION_SCORE_RATIO = 0.85
MIN_CHUNKS_PER_REFERENCED_ACT = 2


def prioritize_referenced_acts[ItemT: RetrievalDocument | tuple[RetrievalDocument, float]](
    items: list[ItemT],
    act_numbers: tuple[str, ...],
    top_n: int,
) -> list[ItemT]:
    """Prefer evidence from acts named in the question within the top_n window.

    Only already eligible (authorized, threshold-passing) items are reordered;
    nothing is added. Every named act that has eligible evidence keeps up to
    ``MIN_CHUNKS_PER_REFERENCED_ACT`` items inside the window. Slots come from
    the weakest items of unnamed acts first, then from named acts holding more
    than their guaranteed share. Inside the window, named-act items come first
    and the reranker order is otherwise kept.
    """
    if not act_numbers or top_n <= 0:
        return items

    def act_of(item: ItemT) -> str | None:
        doc = item[0] if isinstance(item, tuple) else item
        return referenced_act(doc.metadata, act_numbers)

    acts = [act_of(item) for item in items]

    if not any(acts):
        return items

    window = list(range(min(top_n, len(items))))
    outside = list(range(len(window), len(items)))

    for number in act_numbers:
        missing = MIN_CHUNKS_PER_REFERENCED_ACT - sum(acts[i] == number for i in window)
        promoted = [i for i in outside if acts[i] == number][: max(0, missing)]

        for index in promoted:
            replaceable = [i for i in reversed(window) if acts[i] is None] or [
                i
                for i in reversed(window)
                if acts[i] != number
                and sum(acts[j] == acts[i] for j in window) > MIN_CHUNKS_PER_REFERENCED_ACT
            ]

            if not replaceable:
                break

            window.remove(replaceable[0])
            outside.append(replaceable[0])
            outside.remove(index)
            window.append(index)

    window.sort()
    outside.sort()
    named = [i for i in window if acts[i] is not None]
    other = [i for i in window if acts[i] is None]
    return [items[i] for i in (*named, *other, *outside)]


def provision_identity(doc: RetrievalDocument) -> tuple[object, ...] | None:
    metadata = doc.metadata
    section = metadata.get("section") or metadata.get("heading")
    source = metadata.get("document_id") or metadata.get("source")

    if not section or not source or metadata.get("table_id"):
        return None

    parts = section.replace(" \u203a ", " > ").split(" > ")

    # Legacy chunks included the fallback level in their section path.
    # Keep recognizing them until their documents have been fully reindexed.
    if parts[-1] == SENTENCE_UNIT_KIND:
        parts = parts[:-1]

    return (source, metadata.get("act_version_id"), tuple(parts)) if parts else None


def same_evidence_block(anchor: RetrievalDocument, candidate: RetrievalDocument) -> bool:
    """Recognize bounded continuations without crossing provisions or versions."""
    key = provision_identity(anchor)

    if key != provision_identity(candidate):
        return False

    start = anchor.metadata.get("chunk_index")
    end = candidate.metadata.get("chunk_index")

    if key is None:
        # Plain text has no provision label. Only immediate chunks within the
        # same document, edition and inherited conditions form a neighborhood.
        source = anchor.metadata.get("document_id") or anchor.metadata.get("source")
        other_source = candidate.metadata.get("document_id") or candidate.metadata.get("source")
        return bool(
            source
            and source == other_source
            and anchor.metadata.get("act_version_id") == candidate.metadata.get("act_version_id")
            and not anchor.metadata.get("table_id")
            and not candidate.metadata.get("table_id")
            and anchor.metadata.get("content_type") != PageContentType.TABLE.value
            and candidate.metadata.get("content_type") != PageContentType.TABLE.value
            and anchor.metadata.get("parent_units") == candidate.metadata.get("parent_units")
            and start is not None
            and end is not None
            and abs(end - start) <= 1
        )

    return start is None or end is None or abs(end - start) < MAX_PROVISION_CHUNKS


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
    """Merge repeated evidence only within the same source, edition and scope."""
    seen: set[tuple[object, ...]] = set()
    unique_docs: list[DocumentT] = []

    for doc in docs:
        identity = evidence_identity(doc)

        if identity not in seen:
            seen.add(identity)
            unique_docs.append(doc)

    return unique_docs


def evidence_identity(doc: RetrievalDocument) -> tuple[object, ...]:
    """Keep every interpretation field, including complete parent conditions."""
    metadata = doc.metadata
    document_id = metadata.get("document_id")
    source = (
        ("document", document_id)
        if document_id is not None
        else ("source", metadata.get("source") or metadata.get("filename"))
    )
    scope = {key: value for key, value in metadata.items() if key in CHUNK_CONTEXT_FIELDS}
    return (
        source,
        metadata.get("act_id"),
        metadata.get("act_version_id"),
        content_hash(doc.page_content),
        content_hash(json.dumps(scope, sort_keys=True, ensure_ascii=False, default=str)),
    )
