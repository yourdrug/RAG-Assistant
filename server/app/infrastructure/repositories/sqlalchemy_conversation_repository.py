"""SQLAlchemy implementation of ConversationRepository."""

from __future__ import annotations

from domain.entities.conversation import Conversation
from sqlalchemy import text
from sqlalchemy.orm import Session


class SQLAlchemyConversationRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, user_id: int) -> Conversation:
        result = self._db.execute(
            text("INSERT INTO conversations (user_id) VALUES (:uid) RETURNING id"),
            {"uid": user_id},
        )
        conv_id = result.scalar()
        return Conversation(id=conv_id, user_id=user_id)

    def get_by_id(self, conversation_id: int) -> Conversation | None:
        row = self._db.execute(
            text("SELECT id, user_id, created_at FROM conversations WHERE id = :id"),
            {"id": conversation_id},
        ).fetchone()
        if row is None:
            return None
        return Conversation(id=row.id, user_id=row.user_id, created_at=row.created_at)

    def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation:
        if conversation_id:
            row = self._db.execute(
                text("SELECT id FROM conversations WHERE id = :id AND user_id = :uid"),
                {"id": conversation_id, "uid": user_id},
            ).fetchone()
            if row:
                return Conversation(id=conversation_id, user_id=user_id)
        return self.create(user_id)

    def get_owner_id(self, conversation_id: int) -> int | None:
        row = self._db.execute(
            text("SELECT user_id FROM conversations WHERE id = :id"),
            {"id": conversation_id},
        ).fetchone()
        return row.user_id if row else None
