"""RAG post-processing — relevance gate, rejection, temporal conflicts, neighbor enrichment.

Extracted from RagService to isolate post-processing logic from pipeline orchestration.
"""

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date as _date
import logging
import re
import time
from typing import Any

from langchain_core.documents import Document, Document as LCDocument

from application.ports.ml_clients import MLClientPort
from application.rag.stream_events import SourcesEvent, StreamEvent, TextChunk
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import TABLE_CONTEXT_MAX_CHUNKS, ChunkRecord
from domain.knowledge.document_nodes import PARENT_NODE_ID_KEY
from domain.knowledge.enums import PageContentType
from domain.rag.breadth import Breadth
from domain.rag.evidence.table_context import requests_table_overview
from domain.rag.ports import ChunkRetrievalRepository
from domain.rag.ranking import evidence_identity, same_evidence_block, same_text_run
from domain.rag.settings import RagSettings
from infrastructure.metrics.metrics import (
    RAG_RELEVANCE_GATE_TOTAL,
    RAG_STAGE_DURATION,
    RAG_TEMPORAL_VERSION_CONFLICT_TOTAL,
    record_rag_answer,
)
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from infrastructure.ml.rag.context_rendering import render_selected_docs
from infrastructure.ml.rag.context_selection import select_context
from infrastructure.ml.rag.prompt_blocks import NOT_FOUND_ANSWER, is_not_found_answer
from infrastructure.ml.rag.prompt_budget import estimate_text_tokens
from infrastructure.ml.rag.rag_relevance import check_relevance, filter_cited_sources
from infrastructure.ml.rag.table_context import focus_table_rows
from infrastructure.resilience import circuit_breaker


log = logging.getLogger("default")

# estimate_text_tokens counts UTF-8 bytes / 3: one Cyrillic character ≈ 2/3 token.
PARENT_CHARS_PER_TOKEN = 1.5


async def handle_relevance_gate(
    query_for_search: str,
    docs: list[tuple[Document, float]],
    breadth: Breadth,
    rag: RagSettings,
    ml_clients: MLClientPort,
    *,
    context: str | None = None,
) -> bool:
    """Check relevance gate. Returns True if relevant, False if rejected."""
    del breadth

    if not rag.features.relevance_gate_enabled:
        return True

    t0 = time.monotonic()

    breaker = circuit_breaker.get_breaker("llm_auxiliary")
    breaker.check_open()

    async with ml_clients.auxiliary_semaphore:
        try:
            is_relevant, reason = await check_relevance(ml_clients, query_for_search, docs, context=context)
        except Exception:
            await breaker.report_failure()
            raise
        await breaker.report_success()
    RAG_STAGE_DURATION.labels("relevance_gate").observe(time.monotonic() - t0)

    if not is_relevant:
        RAG_RELEVANCE_GATE_TOTAL.labels(result="rejected").inc()
        log.info("Relevance gate: rejected (reason_chars=%d)", len(reason))
        return False

    RAG_RELEVANCE_GATE_TOTAL.labels(result="passed").inc()
    return True


async def reject_not_relevant(
    breadth: Breadth,
    docs: list[tuple[Document, float]],
    avg_sim: float,
    t_pipeline_start: float,
) -> AsyncIterator[StreamEvent]:
    """Yield a rejection message when relevance gate fails."""
    yield TextChunk(text=NOT_FOUND_ANSWER)
    record_rag_answer(
        breadth=breadth.value,
        answer="",
        retrieved_count=len(docs),
        avg_similarity=avg_sim,
    )
    RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)
    yield SourcesEvent(sources=[], confidence=None)


def resolve_temporal_conflicts(docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """Keep all retrieved chunks from the newest eligible version of each act."""
    versions_by_act: dict[int, list[tuple[Document, float, _date]]] = {}
    non_versioned: list[tuple[Document, float]] = []

    for doc, score in docs:
        act_id = doc.metadata.get("act_id")
        act_version_id = doc.metadata.get("act_version_id")

        if act_id is None or act_version_id is None:
            non_versioned.append((doc, score))
            continue

        effective_from = effective_from_for(doc)

        versions_by_act.setdefault(act_id, []).append((doc, score, effective_from))

    resolved = list(non_versioned)

    for act_id, versions in versions_by_act.items():
        if len(versions) == 1:
            resolved.append(versions[0][:2])
            continue

        versions_by_id: dict[int, list[tuple[Document, float, _date]]] = {}

        for item in versions:
            version_id = item[0].metadata["act_version_id"]
            versions_by_id.setdefault(version_id, []).append(item)

        if len(versions_by_id) == 1:
            resolved.extend((doc, score) for doc, score, _ in versions)
            continue

        best_version = max(
            versions_by_id.values(),
            key=lambda chunks: (chunks[0][2], chunks[0][0].metadata["act_version_id"]),
        )
        resolved.extend((doc, score) for doc, score, _ in best_version)
        dropped = len(versions) - len(best_version)
        RAG_TEMPORAL_VERSION_CONFLICT_TOTAL.inc()
        log.warning(
            "Temporal conflict act_id=%d: kept version with effective_from=%s, dropped %d older versions",
            act_id,
            best_version[0][2],
            dropped,
        )

    return resolved


def apply_citation_filter(
    rag: RagSettings, full_answer: str, sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Filter sources to only those cited in the LLM answer."""
    if not rag.features.citation_filter_enabled:
        return sources

    filtered = filter_cited_sources(full_answer, sources)

    if len(filtered) < len(sources):
        log.info(
            "Citation filter: kept %d/%d sources",
            len(filtered),
            len(sources),
        )

    return filtered


@dataclass
class ParentScopes:
    """Parent scopes already expanded in one request and the size one may have."""

    max_chars: int
    seen: set[tuple[int, int | None, int, int]] = field(default_factory=set)


async def fetch_context_neighbors(  # noqa: PLR0913 - one request scope per expansion kind
    doc: Document,
    chunk_search: ChunkRetrievalRepository,
    user: UserContext | None,
    *,
    full_table: bool,
    enumerate_cases: bool,
    hashes: set[str],
    scopes: set[tuple[int, str, int | None, int | None]],
    parents: ParentScopes | None = None,
) -> list[ChunkRecord]:
    """Resolve one authorized table/row scope, a parent scope or a bounded text neighborhood."""
    document_id = doc.metadata.get("document_id")
    chunk_index = doc.metadata.get("chunk_index")

    if user is None or not document_id or chunk_index is None:
        return []

    if doc.metadata.get("content_type") != PageContentType.TABLE.value:
        parent = await fetch_parent_scope(doc, chunk_search, user, parents)

        if parent is not None:
            return parent

        plain_text = not (doc.metadata.get("section") or doc.metadata.get("heading"))
        neighbors = await chunk_search.get_neighbors(
            document_id,
            chunk_index,
            window=3 if enumerate_cases and not plain_text else 1,
            exclude_hashes=hashes,
            user=user,
        )
        return (
            neighbors
            if enumerate_cases and not plain_text
            else [
                neighbor
                for neighbor in neighbors
                if (same_text_run if plain_text else same_evidence_block)(
                    doc, LCDocument(page_content=neighbor.content, metadata=chunk_result_metadata(neighbor))
                )
            ]
        )

    return await fetch_table_context(doc, chunk_search, user, hashes, scopes, full_table=full_table)


async def fetch_parent_scope(
    doc: Document,
    chunk_search: ChunkRetrievalRepository,
    user: UserContext,
    parents: ParentScopes | None,
) -> list[ChunkRecord] | None:
    """Chunks of the largest fitting parent, nearest to the anchor first.

    ``None`` keeps the legacy neighbour heuristics (chunks indexed before parent nodes, or no
    parent fits). Children of one parent fetch it once; the renderer joins them in one block.
    """
    node_id = doc.metadata.get(PARENT_NODE_ID_KEY)

    if parents is None or not node_id or parents.max_chars <= 0:
        return None

    document_id = doc.metadata["document_id"]
    act_version_id = doc.metadata.get("act_version_id")
    chunk_index = doc.metadata["chunk_index"]
    records = await chunk_search.get_parent_context(
        document_id, node_id, user=user, act_version_id=act_version_id, max_chars=parents.max_chars
    )

    if not records:
        return None

    # Records are ordered by chunk_index; their range identifies the selected ancestor.
    scope = (document_id, act_version_id, records[0].chunk_index, records[-1].chunk_index)

    if scope in parents.seen:
        return []

    parents.seen.add(scope)
    siblings = [record for record in records if record.chunk_index != chunk_index]
    return sorted(siblings, key=lambda record: (abs(record.chunk_index - chunk_index), record.chunk_index))


async def enrich_with_neighbors(
    docs: list[tuple[Document, float]],
    *,
    enumerate_cases: bool,
    chunk_search: ChunkRetrievalRepository | None,
    max_context_tokens: int = 6000,
    user: UserContext | None = None,
    query: str = "",
) -> list[tuple[Document, float]]:
    """Expand table rows within their table; full tables require an overview request."""
    full_table = requests_table_overview(query)

    if not full_table:
        docs = [
            (focus_table_rows(doc, query), score)
            if doc.metadata.get("content_type") == PageContentType.TABLE.value
            else (doc, score)
            for doc, score in docs
        ]

    if not chunk_search:
        return docs

    # Select anchors before expansion and reserve half the context for their
    # continuations. Keep the original candidate pool for the final prompt's
    # exact token counter, which may have a different remaining budget.
    anchors = select_neighbor_anchors(docs, max_context_tokens, query)
    existing = {evidence_identity(doc) for doc, _ in docs}
    scopes: set[tuple[int, str, int | None, int | None]] = set()
    # A parent must fit the expansion half of the context; packing below shares it fairly.
    parents = ParentScopes(int(max_context_tokens // 2 * PARENT_CHARS_PER_TOKEN))
    queues: list[tuple[float, deque[ChunkRecord]]] = []

    for doc, score in anchors:
        try:
            neighbors = await fetch_context_neighbors(
                doc,
                chunk_search,
                user,
                full_table=full_table,
                enumerate_cases=enumerate_cases,
                # Equal bodies can carry different conditions even within one
                # document; deduplicate complete evidence after fetching it.
                hashes=set(),
                scopes=scopes,
                parents=parents,
            )
        except Exception:
            log.warning(
                "Failed to fetch context for doc_id=%s", doc.metadata.get("document_id"), exc_info=True
            )
            continue

        queues.append((score, deque(neighbors)))

    expanded = pack_context_neighbors(
        anchors, queues, existing, max_context_tokens, full_table=full_table, query=query
    )
    return docs + expanded[len(anchors) :]


def select_neighbor_anchors(
    docs: list[tuple[Document, float]], max_context_tokens: int, query: str
) -> list[tuple[Document, float]]:
    """Reserve expansion space without excluding a large, stronger anchor."""
    for item in sorted(docs, key=lambda item: -item[1]):
        cost = estimate_text_tokens(render_selected_docs([item])[0])

        if cost <= max_context_tokens:
            anchor_budget = max(1, max_context_tokens // 2, cost)
            _, selected = select_context(docs, anchor_budget, estimate_text_tokens, [query])
            selected_keys = {evidence_identity(doc) for doc, _ in selected}
            return [item for item in docs if evidence_identity(item[0]) in selected_keys]

    return []


def pack_context_neighbors(
    docs: list[tuple[Document, float]],
    queues: list[tuple[float, deque[ChunkRecord]]],
    existing: set[tuple[object, ...]],
    max_context_tokens: int,
    *,
    full_table: bool,
    query: str,
) -> list[tuple[Document, float]]:
    new_docs: list[tuple[Document, float]] = []

    # Give each retrieved scope a turn before adding its second neighbor.
    while any(queue for _, queue in queues):
        for score, queue in queues:
            if not queue:
                continue

            neighbor = queue.popleft()

            neighbor_doc = LCDocument(
                page_content=neighbor.content,
                metadata=chunk_result_metadata(neighbor),
            )

            neighbor_doc = neighbor_doc if full_table else focus_table_rows(neighbor_doc, query)
            identity = evidence_identity(neighbor_doc)

            if identity in existing:
                continue

            trial = [*docs, *new_docs, (neighbor_doc, score * 0.9)]

            if estimate_text_tokens(render_selected_docs(trial)[0]) > max_context_tokens:
                continue

            new_docs.append((neighbor_doc, score * 0.9))
            existing.add(identity)

    return docs + new_docs


def mark_partial_answer(answer: str) -> str:
    """Replace a standalone refusal appended to a substantive answer."""
    if is_not_found_answer(answer):
        return answer

    return re.sub(
        r"(?:^|(?<=[.!?])\s+|\n+)информация не найдена в документах(?:\.(?=\s|$)|$)",
        "\n\nОтвет частичный: остальная информация не подтверждена документами.",
        answer,
        flags=re.IGNORECASE,
    )


def effective_from_for(doc: Document) -> _date:
    effective_from = doc.metadata.get("effective_from") or _date.min

    if isinstance(effective_from, str):
        try:
            effective_from = _date.fromisoformat(effective_from)
        except ValueError:
            effective_from = _date.min

    return effective_from


async def fetch_table_context(
    doc: Document,
    chunk_search: ChunkRetrievalRepository,
    user: UserContext,
    hashes: set[str],
    scopes: set[tuple[int, str, int | None, int | None]],
    *,
    full_table: bool,
) -> list[ChunkRecord]:
    document_id = doc.metadata["document_id"]
    table_id = doc.metadata.get("table_id")
    row_start = None if full_table else doc.metadata.get("table_row_start")
    row_end = None if full_table else doc.metadata.get("table_row_end")

    # Legacy chunks cannot establish safe boundaries. Reindex them.
    if not table_id or (not full_table and (row_start is None or row_end is None)):
        return []

    scope = (document_id, table_id, row_start, row_end)

    if scope in scopes:
        return []

    scopes.add(scope)
    return await chunk_search.get_table_batches(
        document_id,
        table_id,
        exclude_hashes=hashes,
        user=user,
        limit=TABLE_CONTEXT_MAX_CHUNKS,
        row_start=row_start,
        row_end=row_end,
    )
