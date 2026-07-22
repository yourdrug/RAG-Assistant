"""Conversation Repository interface."""

from __future__ import annotations

from typing import Protocol

from domain.entities.conversation import Conversation


class ConversationRepository(Protocol):
    def create(self, user_id: int) -> Conversation: ...
    def get_by_id(self, conversation_id: int) -> Conversation | None: ...
    def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation: ...
    def get_owner_id(self, conversation_id: int) -> int | None: ...
