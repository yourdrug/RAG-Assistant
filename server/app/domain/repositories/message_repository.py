"""Message Repository interface."""

from __future__ import annotations

from typing import Protocol

from domain.entities.message import Message


class MessageRepository(Protocol):
    def save(self, message: Message) -> None: ...
    def get_history(self, conversation_id: int, window: int = 8) -> list[Message]: ...
