"""SQLAlchemy ORM implementation of ConversationRepository."""

from __future__ import annotations

from domain.entities.conversation import Conversation
from domain.repositories.conversation_repository import ConversationListItem
from sqlalchemy import Integer, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ConversationModel, MessageModel


class SQLAlchemyConversationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, user_id: int) -> Conversation:
        orm = ConversationModel(user_id=user_id)
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        return Conversation(id=orm.id, user_id=orm.user_id, creation_date=orm.creation_date)

    async def get_by_id(self, conversation_id: int) -> Conversation | None:
        result = await self._db.execute(
            select(ConversationModel).where(ConversationModel.id == conversation_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        return Conversation(
            id=orm.id, user_id=orm.user_id, creation_date=orm.creation_date, summary=orm.summary
        )

    async def get(self, conversation_id: int) -> Conversation | None:
        return await self.get_by_id(conversation_id)

    async def get_for_update(self, conversation_id: int) -> Conversation | None:
        result = await self._db.execute(
            select(ConversationModel).where(ConversationModel.id == conversation_id).with_for_update()
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        return Conversation(
            id=orm.id, user_id=orm.user_id, creation_date=orm.creation_date, summary=orm.summary
        )

    async def save(self, conversation: Conversation) -> Conversation:
        if conversation.id is not None:
            result = await self._db.execute(
                select(ConversationModel).where(ConversationModel.id == conversation.id)
            )
            orm = result.scalar_one_or_none()
            if orm is not None:
                orm.user_id = conversation.user_id
                await self._db.flush()
                return conversation
        orm = ConversationModel(user_id=conversation.user_id)
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        conversation.id = orm.id
        return conversation

    async def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation:
        if conversation_id:
            conv = await self.get_by_id(conversation_id)
            if conv and conv.is_owned_by(user_id):
                return conv
        return await self.create(user_id)

    async def get_owner_id(self, conversation_id: int) -> int | None:
        result = await self._db.execute(
            select(ConversationModel.user_id).where(ConversationModel.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def update_summary(self, conversation_id: int, summary: str | None) -> None:
        result = await self._db.execute(
            select(ConversationModel).where(ConversationModel.id == conversation_id)
        )
        orm = result.scalar_one_or_none()
        if orm is not None:
            orm.summary = summary
            await self._db.flush()

    async def update_summary_if_unchanged(
        self, conversation_id: int, expected_summary: str | None, new_summary: str | None
    ) -> bool:
        """Optimistic compare-and-swap on the conversation summary.

        Applies ``new_summary`` only if the current stored summary still equals
        ``expected_summary``, preventing a lost update between the read (before
        the external LLM call) and this write. Returns True if applied.
        """
        from sqlalchemy import ColumnElement, update

        expected_clause: ColumnElement[bool]
        if expected_summary is None:
            expected_clause = ConversationModel.summary.is_(None)
        else:
            expected_clause = ConversationModel.summary == expected_summary

        stmt = (
            update(ConversationModel)
            .where(ConversationModel.id == conversation_id)
            .where(expected_clause)
            .values(summary=new_summary)
        )
        result = await self._db.execute(stmt)
        return result.rowcount == 1

    async def clear_summaries_referencing(self, document_id: int) -> int:
        """Reset summaries for conversations whose assistant messages cite the given document.

        Queries the JSON ``sources`` column on assistant messages to find
        conversations that referenced the deleted document, then NULLs their
        summary.  Conversations without a matching citation are left untouched.
        Returns the number of affected rows.

        Edge case: legacy messages where ``sources`` is NULL or lacks
        ``document_id`` will not match — those summaries stay stale but
        are harmless (summary only affects condensation, not retrieval).
        """
        from sqlalchemy import func, text

        cited_conv_ids = (
            select(MessageModel.conversation_id)
            .where(MessageModel.role == "assistant")
            .where(
                func.json_array_elements(func.coalesce(MessageModel.sources, text("'[]'::json")))
                .op("->>")("document_id")
                .cast(Integer)
                == document_id
            )
            .distinct()
            .subquery()
        )

        stmt = (
            update(ConversationModel)
            .where(ConversationModel.summary.isnot(None))
            .where(ConversationModel.id.in_(select(cited_conv_ids.c.conversation_id)))
            .values(summary=None)
        )
        result = await self._db.execute(stmt)
        await self._db.flush()
        return result.rowcount

    async def list_by_user(
        self, user_id: int, limit: int = 50, offset: int = 0
    ) -> list[ConversationListItem]:
        min_msg_ids = (
            select(
                MessageModel.conversation_id,
                func.min(MessageModel.id).label("min_id"),
            )
            .where(MessageModel.role == "user")
            .group_by(MessageModel.conversation_id)
            .subquery()
        )

        first_msg_sq = (
            select(
                min_msg_ids.c.conversation_id,
                MessageModel.content.label("first_content"),
            )
            .join(MessageModel, MessageModel.id == min_msg_ids.c.min_id)
            .subquery()
        )

        msg_count_sq = (
            select(
                MessageModel.conversation_id,
                func.count().label("msg_count"),
            )
            .group_by(MessageModel.conversation_id)
            .subquery()
        )

        result = await self._db.execute(
            select(
                ConversationModel.id,
                ConversationModel.user_id,
                ConversationModel.creation_date,
                first_msg_sq.c.first_content,
                func.coalesce(msg_count_sq.c.msg_count, 0).label("message_count"),
            )
            .outerjoin(first_msg_sq, ConversationModel.id == first_msg_sq.c.conversation_id)
            .outerjoin(msg_count_sq, ConversationModel.id == msg_count_sq.c.conversation_id)
            .where(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.creation_date.desc())
            .limit(limit)
            .offset(offset)
        )

        return [
            ConversationListItem(
                id=row.id,
                user_id=row.user_id,
                creation_date=row.creation_date,
                title=row.first_content,
                message_count=row.message_count,
            )
            for row in result.all()
        ]
