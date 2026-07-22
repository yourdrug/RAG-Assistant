"""Message Entity — child of Conversation aggregate."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from domain.value_objects.message_role import MessageRole


@dataclass
class Message:
    id: int | None = None
    conversation_id: int = 0
    role: MessageRole = MessageRole.USER
    content: str = ""
    sources: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)
