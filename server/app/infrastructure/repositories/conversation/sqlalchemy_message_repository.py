"""SQLAlchemy ORM implementation of MessageRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from domain.chat.message import Message, MessageRole
from infrastructure.database.models import MessageModel


if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession


class SQLAlchemyMessageRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save(self, message: Message) -> None:
        orm = MessageModel(
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            sources=message.sources,
        )
        self._db.add(orm)
        await self._db.flush()
        message.id = orm.id

    async def get_by_id(self, message_id: int) -> Message | None:
        orm = await self._db.get(MessageModel, message_id)

        if orm is None or orm.conversation_id is None:
            return None

        return Message(
            id=orm.id,
            conversation_id=orm.conversation_id,
            role=MessageRole(orm.role),
            content=orm.content,
            sources=orm.sources or [],
            creation_date=orm.creation_date,
        )

    async def get_history(self, conversation_id: int, window: int = 8) -> list[Message]:
        result = await self._db.execute(
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.creation_date.desc())
            .limit(window)
        )
        rows = result.scalars().all()

        return [
            Message(
                id=orm.id,
                role=MessageRole(orm.role),
                content=orm.content,
                sources=orm.sources or [],
                creation_date=orm.creation_date,
            )
            for orm in reversed(rows)
        ]

    async def list_page(
        self,
        conversation_id: int,
        limit: int,
        before_id: int | None = None,
        created_before: datetime | None = None,
    ) -> list[Message]:
        query = select(MessageModel).where(MessageModel.conversation_id == conversation_id)

        if before_id is not None:
            query = query.where(MessageModel.id < before_id)

        if created_before is not None:
            query = query.where(MessageModel.creation_date < created_before)

        result = await self._db.execute(query.order_by(MessageModel.id.desc()).limit(limit))
        return [
            Message(
                id=orm.id,
                conversation_id=conversation_id,
                role=MessageRole(orm.role),
                content=orm.content,
                sources=orm.sources or [],
                creation_date=orm.creation_date,
            )
            for orm in reversed(result.scalars().all())
        ]
