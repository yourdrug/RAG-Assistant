"""Persistence ports for conversations, messages and quality logs."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from domain.chat.chat_log import ChatLog
from domain.chat.conversation import Conversation, ConversationListItem
from domain.chat.message import Message


@runtime_checkable
class ConversationRepository(Protocol):
    async def create(self, user_id: int) -> Conversation: ...
    async def get_for_update(self, conversation_id: int) -> Conversation | None: ...
    async def get_by_id(self, conversation_id: int) -> Conversation | None: ...
    async def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation: ...
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
    async def get_history(self, conversation_id: int, window: int = 8) -> list[Message]: ...

    async def list_page(
        self, conversation_id: int, limit: int, before_id: int | None = None
    ) -> list[Message]: ...


@runtime_checkable
class ChatLogRepository(Protocol):
    async def save(self, log: ChatLog) -> None: ...

    async def list_logs(
        self,
        *,
        user_id: int | None = None,
        user_ids: list[int] | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatLog]: ...

    async def count_logs(
        self,
        *,
        user_id: int | None = None,
        user_ids: list[int] | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
    ) -> int: ...
