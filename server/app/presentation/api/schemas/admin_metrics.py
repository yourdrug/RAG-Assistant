"""Admin metrics schemas."""

from __future__ import annotations

from pydantic import BaseModel


class MetricsResponse(BaseModel):
    db_pool: dict[str, float]
    qdrant: dict[str, float]
    bm25: dict[str, float]
    ollama: list[dict[str, object]]
    rag: dict[str, object]
    ingestion: dict[str, object]
    http_requests: dict[str, object]
