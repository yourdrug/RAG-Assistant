"""SQLAlchemy implementation of ChatLogRepository."""

from __future__ import annotations

from datetime import datetime

from domain.entities.chat_log import ChatLog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChatLogModel


class SQLAlchemyChatLogRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save(self, log: ChatLog) -> None:
        orm = ChatLogModel(
            user_id=log.user_id,
            conversation_id=log.conversation_id,
            question=log.question,
            answer=log.answer,
            sources=log.sources if log.sources else None,
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

    async def list_logs(
        self,
        *,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatLog]:
        stmt = select(ChatLogModel).order_by(ChatLogModel.creation_date.desc())
        stmt = self._apply_filters(
            stmt, user_id=user_id, domain=domain, date_from=date_from, date_to=date_to, search=search
        )
        stmt = stmt.offset(offset).limit(limit)
        result = await self._db.execute(stmt)
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def count_logs(
        self,
        *,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(ChatLogModel)
        stmt = self._apply_filters(
            stmt, user_id=user_id, domain=domain, date_from=date_from, date_to=date_to, search=search
        )
        result = await self._db.execute(stmt)
        return result.scalar_one()

    @staticmethod
    def _apply_filters(
        stmt,
        *,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
    ):
        if user_id is not None:
            stmt = stmt.where(ChatLogModel.user_id == user_id)
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
