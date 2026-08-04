"""SQLAlchemy ORM implementation of MessageRepository."""

from __future__ import annotations

from domain.entities.message import Message
from domain.value_objects.message_role import MessageRole
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import MessageModel


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

    async def get_history(self, conversation_id: int, window: int = 8) -> list[Message]:
        result = await self._db.execute(
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.creation_date.desc())
            .limit(window)
        )
        rows = result.scalars().all()

        messages = []
        for orm in reversed(rows):
            messages.append(
                Message(
                    id=orm.id,
                    role=MessageRole(orm.role),
                    content=orm.content,
                    sources=orm.sources or [],
                    creation_date=orm.creation_date,
                )
            )
        return messages
