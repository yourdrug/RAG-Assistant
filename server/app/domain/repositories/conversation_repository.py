"""Conversation Repository interface."""

from __future__ import annotations

from typing import Protocol

from domain.entities.conversation import Conversation


class ConversationRepository(Protocol):
    async def create(self, user_id: int) -> Conversation: ...
    async def get_by_id(self, conversation_id: int) -> Conversation | None: ...
    async def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation: ...
    async def get_owner_id(self, conversation_id: int) -> int | None: ...
