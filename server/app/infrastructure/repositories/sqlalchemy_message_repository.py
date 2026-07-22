"""SQLAlchemy implementation of MessageRepository."""

from __future__ import annotations

import json

from domain.entities.message import Message
from domain.value_objects.message_role import MessageRole
from sqlalchemy import text
from sqlalchemy.orm import Session


class SQLAlchemyMessageRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def save(self, message: Message) -> None:
        self._db.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content, sources)
                VALUES (:conv_id, :role, :content, :sources)
            """),
            {
                "conv_id": message.conversation_id,
                "role": message.role,
                "content": message.content,
                "sources": json.dumps(message.sources) if message.sources else None,
            },
        )

    def get_history(self, conversation_id: int, window: int = 8) -> list[Message]:
        rows = self._db.execute(
            text("""
                SELECT role, content, sources FROM messages
                WHERE conversation_id = :conv_id
                ORDER BY created_at DESC
                LIMIT :lim
            """),
            {"conv_id": conversation_id, "lim": window},
        ).fetchall()

        messages = []
        for r in reversed(rows):
            sources = json.loads(r.sources) if r.sources else []
            messages.append(
                Message(
                    role=MessageRole(r.role),
                    content=r.content,
                    sources=sources,
                )
            )
        return messages
