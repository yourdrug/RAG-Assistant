"""Persistence ports for conversations, messages and quality logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable


if TYPE_CHECKING:
    from datetime import datetime

    from domain.chat.chat_log import ChatLog
    from domain.chat.conversation import Conversation, ConversationListItem
    from domain.chat.feedback import MessageFeedback
    from domain.chat.message import Message
    from domain.chat.review import ChatLogReview


@dataclass(frozen=True)
class ChatLogFilter:
    user_id: int | None = None
    user_ids: tuple[int, ...] | None = None
    domain: str | None = None
    space_id: int | None = None
    outcome: str | None = None
    error_code: str | None = None
    model: str | None = None
    breadth: str | None = None
    config_hash: str | None = None
    # Reaction of the asking user to the delivered answer; "none" means not rated.
    rating: Literal["up", "down", "none"] | None = None
    reason: str | None = None
    has_comment: bool | None = None
    # "new" also matches entries never reviewed.
    review_status: str | None = None
    # Triage queue: a negative rating or a not_found/error outcome, still unreviewed.
    needs_review: bool = False
    sort_by: Literal["date", "latency", "confidence"] = "date"
    sort_order: Literal["asc", "desc"] = "desc"
    date_from: datetime | None = None
    date_to: datetime | None = None
    search: str | None = None
    limit: int = 50
    offset: int = 0

    def __post_init__(self) -> None:
        if self.user_ids is not None:
            object.__setattr__(self, "user_ids", tuple(self.user_ids))


DEFAULT_CHAT_LOG_FILTER = ChatLogFilter()


@runtime_checkable
class ConversationRepository(Protocol):
    async def create(self, user_id: int) -> Conversation: ...
    async def get_for_update(self, conversation_id: int) -> Conversation | None: ...
    async def get_by_id(self, conversation_id: int) -> Conversation | None: ...
    async def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation: ...
    async def assign_space(self, conversation_id: int, space_id: int) -> int:
        """Atomically bind an unbound conversation; return its persisted space, including a race winner's."""
        ...

    async def get_owner_id(self, conversation_id: int) -> int | None: ...
    async def save(self, conversation: Conversation) -> Conversation: ...
    async def update_summary(self, conversation_id: int, summary: str | None) -> None: ...
    async def update_summary_if_unchanged(
        self, conversation_id: int, expected_summary: str | None, new_summary: str | None
    ) -> bool: ...
    async def clear_summaries_referencing(self, document_id: int) -> int: ...
    async def list_by_user(
        self, user_id: int, limit: int = 50, offset: int = 0
    ) -> list[ConversationListItem]: ...


@runtime_checkable
class MessageRepository(Protocol):
    async def save(self, message: Message) -> None: ...
    async def get_by_id(self, message_id: int) -> Message | None: ...
    async def get_history(self, conversation_id: int, window: int = 8) -> list[Message]: ...

    async def list_page(
        self,
        conversation_id: int,
        limit: int,
        before_id: int | None = None,
        created_before: datetime | None = None,
    ) -> list[Message]:
        """The last ``limit`` messages matching the bounds, in chronological order."""
        ...


@runtime_checkable
class ChatLogRepository(Protocol):
    async def save_trace(self, log_id: int, data: dict[str, Any]) -> None: ...
    async def get_trace(self, log_id: int) -> dict[str, Any] | None: ...
    async def delete_expired(self, trace_days: int, log_days: int, limit: int = 1000) -> int: ...

    async def save_snapshot(self, config_hash: str, data: dict[str, Any]) -> None: ...
    async def get_snapshot(self, config_hash: str) -> dict[str, Any] | None: ...
    async def get_by_id(self, log_id: int, user_ids: tuple[int, ...] | None = None) -> ChatLog | None: ...

    async def save(self, log: ChatLog) -> None: ...

    async def list_reviews(self, log_ids: list[int]) -> list[ChatLogReview]: ...
    async def lock_review(self, log_id: int) -> ChatLogReview:
        """Create the review row if absent and lock it for the rest of the transaction."""
        ...

    async def save_review(self, review: ChatLogReview) -> None: ...
    async def stats(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> dict[str, Any]: ...

    async def list_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> list[ChatLog]: ...

    async def count_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> int: ...


@runtime_checkable
class MessageFeedbackRepository(Protocol):
    async def upsert(self, feedback: MessageFeedback) -> MessageFeedback:
        """Insert or replace the reaction of ``(message_id, user_id)``; concurrent calls keep one row."""
        ...

    async def delete(self, message_id: int, user_id: int) -> bool: ...
    async def list_for_messages(
        self, message_ids: list[int], user_id: int | None = None
    ) -> list[MessageFeedback]: ...
