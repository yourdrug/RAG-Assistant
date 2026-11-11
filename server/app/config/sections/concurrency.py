"""Process-wide concurrency caps for external calls."""

from __future__ import annotations

from pydantic import BaseModel

from config.ownership import Static


class ConcurrencySection(BaseModel):
    # Caps concurrent Qdrant searches (prevents thundering herd).
    qdrant_search_max_concurrent: Static[int] = 16
    qdrant_write_max_concurrent: Static[int] = 8  # caps concurrent Qdrant upserts (prevents write contention)
    llm_max_concurrent: Static[int] = 16
    llm_generation_max_concurrent: Static[int] = 12
    llm_auxiliary_max_concurrent: Static[int] = 4
    embedding_max_concurrent: Static[int] = 16
    ingestion_max_concurrent: Static[int] = 8
    reranker_max_concurrent: Static[int] = 8
    semaphore_acquire_timeout: Static[int] = 30  # seconds, max wait for semaphore acquire before 429/503
