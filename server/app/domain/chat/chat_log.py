"""ChatLog domain entity — persistent Q&A log for quality tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ChatLog:
    user_id: int | None = None
    conversation_id: int | None = None
    question: str = ""
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    latency_ms: int | None = None
    model_used: str | None = None
    breadth: str | None = None
    domain: str | None = None
    retrieval_count: int | None = None
    reranker_score: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    id: int | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
