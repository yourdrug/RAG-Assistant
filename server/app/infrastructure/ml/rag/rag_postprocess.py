"""RAG post-processing — relevance gate, rejection, temporal conflicts, neighbor enrichment.

Extracted from RagService to isolate post-processing logic from pipeline orchestration.
"""

import logging
import re
import time
from collections.abc import AsyncIterator
from datetime import date as _date

from domain.value_objects.llm_provider import Breadth
from domain.value_objects.not_found_patterns import NOT_FOUND_PATTERNS
from domain.value_objects.rag_settings import RagSettings
from domain.value_objects.stream_events import SourcesEvent, StreamEvent, TextChunk
from infrastructure.metrics.metrics import (
    RAG_RELEVANCE_GATE_TOTAL,
    RAG_STAGE_DURATION,
    RAG_TEMPORAL_VERSION_CONFLICT_TOTAL,
    record_rag_answer,
)
from infrastructure.ml.rag.rag_formatting import CHARS_PER_TOKEN
from infrastructure.ml.rag.rag_relevance import check_relevance, filter_cited_sources

log = logging.getLogger("default")


async def handle_relevance_gate(
    query_for_search: str,
    docs: list,
    breadth: Breadth,
    rag: RagSettings,
    ml_clients,
) -> bool:
    """Check relevance gate. Returns True if relevant, False if rejected."""
    if not rag.features.relevance_gate_enabled:
        return True
    t0 = time.monotonic()

    from infrastructure.resilience.circuit_breaker import get_breaker

    breaker = get_breaker("llm_auxiliary")
    breaker.check_open()

    async with ml_clients.auxiliary_semaphore:
        try:
            is_relevant, reason = await check_relevance(ml_clients, query_for_search, docs)
        except Exception:
            await breaker.report_failure()
            raise
        await breaker.report_success()
    RAG_STAGE_DURATION.labels("relevance_gate").observe(time.monotonic() - t0)
    if not is_relevant:
        RAG_RELEVANCE_GATE_TOTAL.labels(result="rejected").inc()
        log.info("Relevance gate: rejected (%s)", reason)
        return False
    RAG_RELEVANCE_GATE_TOTAL.labels(result="passed").inc()
    return True


async def reject_not_relevant(
    breadth: Breadth,
    docs: list,
    avg_sim: float,
    t_pipeline_start: float,
) -> AsyncIterator[StreamEvent]:
    """Yield a rejection message when relevance gate fails."""
    yield TextChunk(
        text="К сожалению, я не нашёл релевантной информации "
        "в загруженных документах для ответа на ваш вопрос."
    )
    record_rag_answer(
        breadth=breadth.value,
        answer="",
        retrieved_count=len(docs),
        avg_similarity=avg_sim,
    )
    RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)
    yield SourcesEvent(sources=[], confidence=None)


def resolve_temporal_conflicts(docs: list) -> list:
    """Resolve temporal version conflicts by keeping only the latest version per act."""
    versions_by_act: dict[int, list[tuple]] = {}
    non_versioned: list[tuple] = []

    for doc, score in docs:
        act_id = doc.metadata.get("act_id")
        act_version_id = doc.metadata.get("act_version_id")
        if act_id is None or act_version_id is None:
            non_versioned.append((doc, score))
            continue
        effective_from = doc.metadata.get("effective_from") or _date.min
        versions_by_act.setdefault(act_id, []).append((doc, score, effective_from))

    resolved = list(non_versioned)
    for act_id, versions in versions_by_act.items():
        if len(versions) == 1:
            resolved.append(versions[0][:2])
            continue
        best = max(versions, key=lambda x: x[2])
        resolved.append(best[:2])
        dropped = len(versions) - 1
        RAG_TEMPORAL_VERSION_CONFLICT_TOTAL.inc()
        log.warning(
            "Temporal conflict act_id=%d: kept version with effective_from=%s, dropped %d older versions",
            act_id,
            best[2],
            dropped,
        )

    return resolved


def apply_citation_filter(rag: RagSettings, full_answer: str, sources: list[dict]) -> list[dict]:
    """Filter sources to only those cited in the LLM answer."""
    if not rag.features.citation_filter_enabled:
        return sources
    return filter_cited_sources(full_answer, sources)


async def enrich_with_neighbors(
    docs: list[tuple],
    enumerate_cases: bool,
    chunk_search,
    max_context_tokens: int = 6000,
    user: dict | None = None,
    group_ids: list[int] | None = None,
) -> list[tuple]:
    """Add neighboring chunks from the same document for richer context."""
    if not chunk_search or not enumerate_cases:
        return docs

    from langchain.schema import Document as LCDocument

    existing_hashes = {h for h in (doc.metadata.get("content_hash") for doc, _ in docs) if h is not None}
    new_docs: list[tuple] = []
    current_chars = sum(len(doc.page_content) for doc, _ in docs)
    max_chars = max_context_tokens * CHARS_PER_TOKEN

    for doc, score in docs:
        chunk_index = doc.metadata.get("chunk_index")
        document_id = doc.metadata.get("document_id")

        if not document_id or chunk_index is None:
            continue

        is_table = doc.metadata.get("content_type") == "table"
        try:
            if is_table:
                neighbors = await chunk_search.get_table_batches(
                    document_id,
                    chunk_index,
                    exclude_hashes=existing_hashes,
                    user=user,
                    group_ids=group_ids,
                )
            else:
                neighbors = await chunk_search.get_neighbors(
                    document_id,
                    chunk_index,
                    window=3,
                    exclude_hashes=existing_hashes,
                    user=user,
                    group_ids=group_ids,
                )
        except Exception:
            log.warning(
                "Failed to fetch neighbors for doc_id=%s chunk_index=%s",
                document_id,
                chunk_index,
                exc_info=True,
            )
            continue

        for n in neighbors:
            if n.content_hash and n.content_hash not in existing_hashes:
                neighbor_chars = len(n.content)
                if current_chars + neighbor_chars > max_chars:
                    break
                neighbor_doc = LCDocument(
                    page_content=n.content,
                    metadata={
                        "source": n.filename,
                        "document_id": n.document_id,
                        "content_hash": n.content_hash,
                        "chunk_index": n.chunk_index,
                        "chunk_id": n.chunk_id,
                        "section": n.section,
                        "heading": n.heading,
                        "heading_level": n.heading_level,
                        "content_type": n.content_type,
                        "doc_title": n.doc_title,
                        "doc_type": n.doc_type,
                    },
                )
                new_docs.append((neighbor_doc, score * 0.9))
                existing_hashes.add(n.content_hash)
                current_chars += neighbor_chars

    return docs + new_docs


def is_not_found_answer(answer: str) -> bool:
    """Return True if the LLM answer indicates information was not found."""
    lower = answer.lower().strip()
    if "информация не найдена в документах" in lower:
        return True
    if len(lower) < 50 and not re.search(r"\[\d+\]", answer):
        return any(p in lower for p in NOT_FOUND_PATTERNS)
    return False
