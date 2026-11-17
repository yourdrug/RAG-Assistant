"""SQLAlchemy persistence for answer attempts and non-secret configuration snapshots."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from domain.chat.chat_log import ChatLog
from domain.chat.repositories import DEFAULT_CHAT_LOG_FILTER
from domain.rag.outcome import AnswerOutcome
from infrastructure.database.models import ChatLogModel, chat_log_traces, rag_config_snapshots


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select

    from domain.chat.repositories import ChatLogFilter


class SQLAlchemyChatLogRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save(self, log: ChatLog) -> None:
        data = {field.name: getattr(log, field.name) for field in fields(ChatLog) if field.name != "id"}
        orm = ChatLogModel(**data)
        self._db.add(orm)
        await self._db.flush()
        log.id = orm.id

    async def save_trace(self, log_id: int, data: dict[str, Any]) -> None:
        await self._db.execute(
            insert(chat_log_traces).values(
                chat_log_id=log_id, schema_version=data["schema_version"], data=data
            )
        )

    async def get_trace(self, log_id: int) -> dict[str, Any] | None:
        result = await self._db.execute(
            select(chat_log_traces.c.data).where(chat_log_traces.c.chat_log_id == log_id)
        )
        return result.scalar_one_or_none()

    async def delete_expired(self, trace_days: int, log_days: int, limit: int = 1000) -> int:
        now = datetime.now(UTC)
        trace_ids = (
            select(chat_log_traces.c.chat_log_id)
            .join(ChatLogModel, ChatLogModel.id == chat_log_traces.c.chat_log_id)
            .where(ChatLogModel.creation_date < now - timedelta(days=trace_days))
            .limit(limit)
        )
        traces = await self._db.execute(
            delete(chat_log_traces)
            .where(chat_log_traces.c.chat_log_id.in_(trace_ids))
            .returning(chat_log_traces.c.chat_log_id)
        )
        log_ids = (
            select(ChatLogModel.id)
            .where(ChatLogModel.creation_date < now - timedelta(days=log_days))
            .limit(limit)
        )
        logs = await self._db.execute(
            delete(ChatLogModel).where(ChatLogModel.id.in_(log_ids)).returning(ChatLogModel.id)
        )
        return len(traces.all()) + len(logs.all())

    async def save_snapshot(self, config_hash: str, data: dict[str, Any]) -> None:
        await self._db.execute(
            insert(rag_config_snapshots)
            .values(hash=config_hash, data=data)
            .on_conflict_do_nothing(index_elements=["hash"])
        )

    async def get_snapshot(self, config_hash: str) -> dict[str, Any] | None:
        result = await self._db.execute(
            select(rag_config_snapshots.c.data).where(rag_config_snapshots.c.hash == config_hash)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, log_id: int, user_ids: tuple[int, ...] | None = None) -> ChatLog | None:
        stmt = select(ChatLogModel).where(ChatLogModel.id == log_id)

        if user_ids is not None:
            stmt = stmt.where(ChatLogModel.user_id.in_(user_ids))

        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self.to_entity(orm) if orm is not None else None

    async def list_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> list[ChatLog]:
        column = {
            "date": ChatLogModel.creation_date,
            "latency": ChatLogModel.latency_ms,
            "confidence": ChatLogModel.confidence,
        }[filters.sort_by]
        ordering = column.asc() if filters.sort_order == "asc" else column.desc()
        stmt = self.apply_filters(select(ChatLogModel), filters).order_by(
            ordering.nulls_last(), ChatLogModel.id.desc()
        )
        result = await self._db.execute(stmt.offset(filters.offset).limit(filters.limit))
        return [self.to_entity(orm) for orm in result.scalars().all()]

    async def count_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> int:
        result = await self._db.execute(
            self.apply_filters(select(func.count()).select_from(ChatLogModel), filters)
        )
        return result.scalar_one()

    @staticmethod
    def apply_filters[T: tuple[Any, ...]](stmt: Select[T], filters: ChatLogFilter) -> Select[T]:
        for name in ("user_id", "domain", "space_id", "outcome", "breadth", "config_hash"):
            value = getattr(filters, name)

            if value is not None:
                stmt = stmt.where(getattr(ChatLogModel, name) == value)

        if filters.model is not None:
            stmt = stmt.where(ChatLogModel.model_used == filters.model)

        if filters.user_ids is not None:
            stmt = stmt.where(ChatLogModel.user_id.in_(filters.user_ids))

        if filters.date_from is not None:
            stmt = stmt.where(ChatLogModel.creation_date >= filters.date_from)

        if filters.date_to is not None:
            stmt = stmt.where(ChatLogModel.creation_date <= filters.date_to)

        if filters.search:
            pattern = f"%{filters.search}%"
            stmt = stmt.where(ChatLogModel.question.ilike(pattern) | ChatLogModel.answer.ilike(pattern))

        return stmt

    @staticmethod
    def to_entity(orm: ChatLogModel) -> ChatLog:
        data = {field.name: getattr(orm, field.name) for field in fields(ChatLog)}
        data["sources"] = data["sources"] or []
        data["outcome"] = AnswerOutcome(data["outcome"])
        return ChatLog(**data)
