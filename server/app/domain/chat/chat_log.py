"""ChatLog domain entity — persistent Q&A log for quality tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from domain.rag.outcome import AnswerOutcome


if TYPE_CHECKING:
    from datetime import date


@dataclass
class ChatLog:
    outcome: AnswerOutcome = AnswerOutcome.GENERATED
    request_id: str | None = None
    error_code: str | None = None
    candidates_count: int | None = None
    top_rerank_score: float | None = None
    condensed_query: str | None = None
    sub_queries: list[str] | None = None
    refined_query: str | None = None
    aux_tokens: int | None = None
    retrieval_ms: int | None = None
    rerank_ms: int | None = None
    config_hash: str | None = None
    config_version: str | None = None
    evidence_profile: str | None = None
    user_id: int | None = None
    conversation_id: int | None = None
    question: str = ""
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int | None = None
    model_used: str | None = None
    breadth: str | None = None
    domain: str | None = None
    message_id: int | None = None
    space_id: int = 0
    pack_key: str | None = None
    depth_requested: str | None = None
    as_of_date: date | None = None
    confidence: float | None = None
    sources_count: int | None = None
    ttft_ms: int | None = None
    generation_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    id: int | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
