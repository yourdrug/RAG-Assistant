"""Chat-related DTOs -- immutable data-transfer objects for the chat API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict


if TYPE_CHECKING:
    from datetime import datetime

    from application.uow import UnitOfWork
    from domain.entities.message import Message
    from domain.value_objects.stream_events import UsageReport


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: int
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatLogCreateOptions(TypedDict):
    user_id: int
    conv_id: int
    question: str
    full_answer: str
    sources: list[dict[str, Any]]
    latency_ms: int
    depth: str | None
    retrieval_count: int
    reranker_score: float | None
    model_used: NotRequired[str | None]
    domain: NotRequired[str | None]
    input_tokens: NotRequired[int | None]
    output_tokens: NotRequired[int | None]
    _existing_uow: NotRequired[UnitOfWork | None]


class ChatLogListOptions(TypedDict, total=False):
    user_id: int | None
    user_ids: list[int] | None
    domain: str | None
    date_from: datetime | None
    date_to: datetime | None
    search: str | None
    limit: int
    offset: int


class ChatPersistenceOptions(TypedDict):
    conv_id: int
    user_id: int
    question: str
    full_answer: str
    sources: list[dict[str, Any]]
    latency_ms: int
    depth: str | None
    history: list[Message]
    usage: NotRequired[UsageReport | None]
    model_used: NotRequired[str | None]
    domain: NotRequired[str | None]
