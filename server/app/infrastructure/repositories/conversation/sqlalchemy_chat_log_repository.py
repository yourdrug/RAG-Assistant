"""SQLAlchemy implementation of ChatLogRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from domain.chat.chat_log import ChatLog
from domain.chat.repositories import DEFAULT_CHAT_LOG_FILTER
from infrastructure.database.models import ChatLogModel


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select

    from domain.chat.repositories import ChatLogFilter


class SQLAlchemyChatLogRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save(self, log: ChatLog) -> None:
        orm = ChatLogModel(
            user_id=log.user_id,
            conversation_id=log.conversation_id,
            question=log.question,
            answer=log.answer,
            sources=log.sources or None,
            latency_ms=log.latency_ms,
            model_used=log.model_used,
            breadth=log.breadth,
            domain=log.domain,
            retrieval_count=log.retrieval_count,
            reranker_score=log.reranker_score,
            input_tokens=log.input_tokens,
            output_tokens=log.output_tokens,
        )
        self._db.add(orm)
        await self._db.flush()

    async def list_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> list[ChatLog]:
        user_id = filters.user_id
        user_ids = filters.user_ids
        domain = filters.domain
        date_from = filters.date_from
        date_to = filters.date_to
        search = filters.search
        limit = filters.limit
        offset = filters.offset
        stmt = select(ChatLogModel).order_by(ChatLogModel.creation_date.desc())
        stmt = self.apply_filters(
            stmt,
            user_id=user_id,
            user_ids=user_ids,
            domain=domain,
            date_from=date_from,
            date_to=date_to,
            search=search,
        )
        stmt = stmt.offset(offset).limit(limit)
        result = await self._db.execute(stmt)
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def count_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> int:
        user_id = filters.user_id
        user_ids = filters.user_ids
        domain = filters.domain
        date_from = filters.date_from
        date_to = filters.date_to
        search = filters.search
        stmt = select(func.count()).select_from(ChatLogModel)
        stmt = self.apply_filters(
            stmt,
            user_id=user_id,
            user_ids=user_ids,
            domain=domain,
            date_from=date_from,
            date_to=date_to,
            search=search,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one()

    @staticmethod
    def apply_filters[T: tuple[Any, ...]](
        stmt: Select[T],
        *,
        user_id: int | None = None,
        user_ids: Sequence[int] | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
    ) -> Select[T]:
        if user_id is not None:
            stmt = stmt.where(ChatLogModel.user_id == user_id)

        if user_ids is not None:
            stmt = stmt.where(ChatLogModel.user_id.in_(user_ids))

        if domain is not None:
            stmt = stmt.where(ChatLogModel.domain == domain)

        if date_from is not None:
            stmt = stmt.where(ChatLogModel.creation_date >= date_from)

        if date_to is not None:
            stmt = stmt.where(ChatLogModel.creation_date <= date_to)

        if search:
            pattern = f"%{search}%"
            stmt = stmt.where(ChatLogModel.question.ilike(pattern) | ChatLogModel.answer.ilike(pattern))

        return stmt

    @staticmethod
    def _to_entity(orm: ChatLogModel) -> ChatLog:
        return ChatLog(
            id=orm.id,
            creation_date=orm.creation_date,
            user_id=orm.user_id,
            conversation_id=orm.conversation_id,
            question=orm.question,
            answer=orm.answer,
            sources=orm.sources or [],
            latency_ms=orm.latency_ms,
            model_used=orm.model_used,
            breadth=orm.breadth,
            domain=orm.domain,
            retrieval_count=orm.retrieval_count,
            reranker_score=orm.reranker_score,
            input_tokens=orm.input_tokens,
            output_tokens=orm.output_tokens,
        )
