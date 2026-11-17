"""Chat-related DTOs -- immutable data-transfer objects for the chat API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from datetime import date

    from application.rag.stream_events import PipelineMetaEvent, UsageReport
    from domain.chat.message import Message


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: int
    message_id: int | None = None
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
    space_id: int
    pack_key: str | None = None
    message_id: int | None = None
    depth_requested: str | None = None
    as_of_date: date | None = None
    confidence: float | None = None
    metadata: PipelineMetaEvent | None = None
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
    space_id: int = 0
    pack_key: str | None = None
    depth_requested: str | None = None
    as_of_date: date | None = None
    confidence: float | None = None
    metadata: PipelineMetaEvent | None = None
    usage: UsageReport | None = None
    model_used: str | None = None
    domain: str | None = None
