"""Process-wide concurrency caps for external calls."""

from __future__ import annotations

from pydantic import BaseModel, PositiveInt

from config.ownership import Static


class ConcurrencySection(BaseModel):
    # Positive only: a zero cap or timeout makes every acquire time out.
    # Caps concurrent Qdrant searches (prevents thundering herd).
    qdrant_search_max_concurrent: Static[PositiveInt] = 16
    # Caps concurrent Qdrant upserts (prevents write contention).
    qdrant_write_max_concurrent: Static[PositiveInt] = 8
    llm_max_concurrent: Static[PositiveInt] = 16
    llm_generation_max_concurrent: Static[PositiveInt] = 12
    llm_auxiliary_max_concurrent: Static[PositiveInt] = 4
    embedding_max_concurrent: Static[PositiveInt] = 16
    ingestion_max_concurrent: Static[PositiveInt] = 8
    reranker_max_concurrent: Static[PositiveInt] = 8
    # Seconds, max wait for semaphore acquire before 429/503.
    semaphore_acquire_timeout: Static[PositiveInt] = 30
