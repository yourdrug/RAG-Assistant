"""Message repository interface -- persistence for chat message entities."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.message import Message


@runtime_checkable
class MessageRepository(Protocol):
    async def save(self, message: Message) -> None: ...
    async def get_history(self, conversation_id: int, window: int = 8) -> list[Message]: ...
