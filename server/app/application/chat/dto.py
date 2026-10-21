"""Chat-related DTOs -- immutable data-transfer objects for the chat API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from domain.chat.message import Message
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
