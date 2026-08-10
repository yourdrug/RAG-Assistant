"""Conversation entity -- aggregate root for the Chat bounded context.

Groups a sequence of Messages between a user and the RAG assistant.
Tracks title, ownership, and creation timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from domain.entities.message import Message


@dataclass
class Conversation:
    id: int | None = None
    user_id: int = 0
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    messages: list[Message] = field(default_factory=list)

    def is_owned_by(self, user_id: int) -> bool:
        return self.user_id == user_id

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
