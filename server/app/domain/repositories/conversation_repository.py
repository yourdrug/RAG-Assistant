"""Conversation Repository interface."""

from __future__ import annotations

from typing import Protocol

from domain.entities.conversation import Conversation


class ConversationListItem:
    def __init__(
        self, id: int, user_id: int, creation_date, title: str | None = None, message_count: int = 0
    ):
        self.id = id
        self.user_id = user_id
        self.creation_date = creation_date
        self.title = title
        self.message_count = message_count


class ConversationRepository(Protocol):
    async def create(self, user_id: int) -> Conversation: ...
    async def get_by_id(self, conversation_id: int) -> Conversation | None: ...
    async def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation: ...
    async def get_owner_id(self, conversation_id: int) -> int | None: ...
    async def list_by_user(
        self, user_id: int, limit: int = 50, offset: int = 0
    ) -> list[ConversationListItem]: ...
