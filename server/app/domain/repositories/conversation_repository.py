"""Conversation repository interface -- CRUD and listing for Conversation aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from domain.entities.conversation import Conversation


@dataclass(frozen=True)
class ConversationListItem:
    id: int
    user_id: int
    creation_date: object
    title: str | None = None
    message_count: int = 0


@runtime_checkable
class ConversationRepository(Protocol):
    async def create(self, user_id: int) -> Conversation: ...
    async def get(self, conversation_id: int) -> Conversation | None: ...
    async def get_for_update(self, conversation_id: int) -> Conversation | None: ...
    async def get_by_id(self, conversation_id: int) -> Conversation | None: ...
    async def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation: ...
    async def get_owner_id(self, conversation_id: int) -> int | None: ...
    async def save(self, conversation: Conversation) -> Conversation: ...
    async def list_by_user(
        self, user_id: int, limit: int = 50, offset: int = 0
    ) -> list[ConversationListItem]: ...
