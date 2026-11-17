"""SQLAlchemy persistence for user reactions to assistant answers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from domain.chat.feedback import MessageFeedback
from infrastructure.database.models import message_feedback


if TYPE_CHECKING:
    from sqlalchemy import RowMapping
    from sqlalchemy.ext.asyncio import AsyncSession


class SQLAlchemyMessageFeedbackRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def upsert(self, feedback: MessageFeedback) -> MessageFeedback:
        values = {
            "rating": feedback.rating.value,
            "reasons": sorted(reason.value for reason in feedback.reasons),
            "comment": feedback.comment,
        }
        inserted = insert(message_feedback).values(
            message_id=feedback.message_id, user_id=feedback.user_id, **values
        )
        # One row per (message, user) even under concurrent requests: the last write wins.
        stmt = inserted.on_conflict_do_update(
            constraint="uq_message_feedback_message_user",
            set_={**values, "update_date": func.now()},
        ).returning(message_feedback)
        row = (await self._db.execute(stmt)).mappings().one()
        return self.to_entity(row)

    async def delete(self, message_id: int, user_id: int) -> bool:
        result = await self._db.execute(
            delete(message_feedback)
            .where(message_feedback.c.message_id == message_id, message_feedback.c.user_id == user_id)
            .returning(message_feedback.c.id)
        )
        return result.first() is not None

    async def list_for_messages(
        self, message_ids: list[int], user_id: int | None = None
    ) -> list[MessageFeedback]:
        if not message_ids:
            return []

        stmt = select(message_feedback).where(message_feedback.c.message_id.in_(message_ids))

        if user_id is not None:
            stmt = stmt.where(message_feedback.c.user_id == user_id)

        rows = (await self._db.execute(stmt)).mappings().all()
        return [self.to_entity(row) for row in rows]

    @staticmethod
    def to_entity(row: RowMapping) -> MessageFeedback:
        return MessageFeedback(
            message_id=row["message_id"],
            user_id=row["user_id"],
            rating=row["rating"],
            reasons=frozenset(row["reasons"] or ()),
            comment=row["comment"],
            creation_date=row["creation_date"],
            update_date=row["update_date"],
        )
