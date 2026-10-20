"""Chat-related DTOs -- immutable data-transfer objects for the chat API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict


if TYPE_CHECKING:
    from datetime import datetime

    from domain.entities.message import Message
    from domain.rag.stream_events import UsageReport


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: int
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, kw_only=True)
class CreateChatLogCommand:
    user_id: int
    conv_id: int
    question: str
    full_answer: str
    sources: list[dict[str, Any]]
    latency_ms: int
    depth: str | None
    retrieval_count: int
    reranker_score: float | None
    model_used: str | None = None
    domain: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatLogListOptions(TypedDict, total=False):
    user_id: int | None
    user_ids: list[int] | None
    domain: str | None
    date_from: datetime | None
    date_to: datetime | None
    search: str | None
    limit: int
    offset: int


@dataclass(frozen=True, kw_only=True)
class PersistChatCommand:
    conv_id: int
    user_id: int
    question: str
    full_answer: str
    sources: list[dict[str, Any]]
    latency_ms: int
    depth: str | None
    history: list[Message]
    usage: UsageReport | None = None
    model_used: str | None = None
    domain: str | None = None
