"""SQLAlchemy ORM implementation of ConversationRepository."""

from __future__ import annotations

from domain.entities.conversation import Conversation
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ConversationModel


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
        return Conversation(id=orm.id, user_id=orm.user_id, creation_date=orm.creation_date)

    async def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation:
        if conversation_id:
            conv = await self.get_by_id(conversation_id)
            if conv and conv.user_id == user_id:
                return conv
        return await self.create(user_id)

    async def get_owner_id(self, conversation_id: int) -> int | None:
        result = await self._db.execute(
            select(ConversationModel.user_id).where(ConversationModel.id == conversation_id)
        )
        return result.scalar_one_or_none()
