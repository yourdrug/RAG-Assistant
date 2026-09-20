"""RAG cache — answer cache operations (check, hit handling, store).

Extracted from RagService to isolate cache I/O from pipeline orchestration.
"""

import logging
import time
from collections.abc import AsyncIterator

from domain.value_objects.llm_provider import Breadth
from domain.value_objects.rag_settings import RagSettings
from domain.value_objects.stream_events import PipelineMetaEvent, SourcesEvent, StreamEvent, TextChunk
from infrastructure.ml.answer_cache import find_cached_answer, store_cached_answer
from infrastructure.metrics.metrics import (
    RAG_CACHE_HITS_TOTAL,
    RAG_CACHE_MISSES_TOTAL,
    RAG_STAGE_DURATION,
    record_rag_answer,
)

log = logging.getLogger("default")


async def check_cache(
    rag: RagSettings,
    q_hash: str,
    vis_hash: str,
) -> dict | None:
    """Look up cached answer. Returns cache entry or None on miss."""
    if not rag.features.cache_enabled:
        return None
    t0 = time.monotonic()
    cached = await find_cached_answer(q_hash, vis_hash, cache_enabled=rag.features.cache_enabled)
    RAG_STAGE_DURATION.labels("cache_lookup").observe(time.monotonic() - t0)
    if cached is None:
        RAG_CACHE_MISSES_TOTAL.inc()
        return None
    return cached


async def handle_cache_hit(
    cached: dict,
    q_hash: str,
    t_pipeline_start: float,
    rag: RagSettings | None = None,
    pii_redactor=None,
) -> AsyncIterator[StreamEvent]:
    """Yield events for a cache hit (answer text + sources)."""
    RAG_CACHE_HITS_TOTAL.inc()
    log.info("Cache hit for question hash=%s", q_hash[:12])
    answer_text = cached["answer"]

    if rag is not None and rag.pii_redaction_enabled and pii_redactor is not None:
        answer_text, pii_found = pii_redactor.scan_and_redact(answer_text)
        if pii_found:
            log.warning("PII detected in cached answer: types=%s", pii_found)

    yield TextChunk(text=answer_text)
    record_rag_answer(breadth=Breadth.NARROW.value, answer=answer_text, retrieved_count=0, avg_similarity=0.0)
    RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)
    yield SourcesEvent(sources=cached["sources"], confidence=None)
    yield PipelineMetaEvent(breadth=None, domain="", ttft_sec=0.0)


async def store_answer_cache(
    docs: list,
    query_for_search: str,
    q_hash: str,
    vis_hash: str,
    full_answer: str,
    sources: list[dict],
    cache_enabled: bool = True,
) -> None:
    """Persist answer to cache for future lookups."""
    doc_ids = []
    for doc, _score in docs:
        did = doc.metadata.get("document_id")
        if did is not None:
            doc_ids.append(did)
    await store_cached_answer(
        question_text=query_for_search,
        question_hash=q_hash,
        answer=full_answer,
        sources=sources,
        visibility_scope_hash=vis_hash,
        document_ids=doc_ids,
        cache_enabled=cache_enabled,
    )
