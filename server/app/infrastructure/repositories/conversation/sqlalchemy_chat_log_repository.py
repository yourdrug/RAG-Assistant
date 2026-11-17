"""SQLAlchemy persistence for answer attempts and non-secret configuration snapshots."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, delete, exists, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from domain.chat.chat_log import ChatLog
from domain.chat.feedback import FeedbackRating
from domain.chat.repositories import DEFAULT_CHAT_LOG_FILTER
from domain.chat.review import ChatLogReview, ReviewStatus
from domain.rag.outcome import AnswerOutcome
from infrastructure.database.models import (
    ChatLogModel,
    chat_log_reviews,
    chat_log_traces,
    message_feedback,
    rag_config_snapshots,
)


if TYPE_CHECKING:
    from sqlalchemy import ColumnElement, SQLColumnExpression
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select

    from domain.chat.repositories import ChatLogFilter


STATS_GROUP_LIMIT = 20
# The reaction of the asking user to the delivered answer of this entry.
FEEDBACK_JOIN = and_(
    message_feedback.c.message_id == ChatLogModel.message_id,
    message_feedback.c.user_id == ChatLogModel.user_id,
)
TRIAGE_OUTCOMES = (AnswerOutcome.NOT_FOUND.value, AnswerOutcome.ERROR.value)


# Filters read their own alias: stats() also joins message_feedback in the outer query,
# and a subquery over the same table would be correlated away entirely.
FILTER_FEEDBACK = message_feedback.alias("filter_feedback")


def feedback_exists(*conditions: ColumnElement[bool]) -> ColumnElement[bool]:
    return (
        exists()
        .where(
            FILTER_FEEDBACK.c.message_id == ChatLogModel.message_id,
            FILTER_FEEDBACK.c.user_id == ChatLogModel.user_id,
            *conditions,
        )
        .correlate(ChatLogModel)
    )


def reviewed() -> ColumnElement[bool]:
    """The entry has a review beyond the initial ``new`` status."""
    return exists().where(
        chat_log_reviews.c.chat_log_id == ChatLogModel.id,
        chat_log_reviews.c.status != ReviewStatus.NEW.value,
    )


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

        return SQLAlchemyChatLogRepository.apply_review_filters(stmt, filters)

    @staticmethod
    def apply_review_filters[T: tuple[Any, ...]](stmt: Select[T], filters: ChatLogFilter) -> Select[T]:
        if filters.rating == "none":
            stmt = stmt.where(~feedback_exists())
        elif filters.rating is not None:
            stmt = stmt.where(feedback_exists(FILTER_FEEDBACK.c.rating == filters.rating))

        if filters.reason is not None:
            stmt = stmt.where(feedback_exists(FILTER_FEEDBACK.c.reasons.contains([filters.reason])))

        if filters.has_comment is not None:
            commented = feedback_exists(FILTER_FEEDBACK.c.comment.is_not(None))
            stmt = stmt.where(commented if filters.has_comment else ~commented)

        if filters.review_status == ReviewStatus.NEW:
            stmt = stmt.where(~reviewed())
        elif filters.review_status is not None:
            stmt = stmt.where(
                exists().where(
                    chat_log_reviews.c.chat_log_id == ChatLogModel.id,
                    chat_log_reviews.c.status == filters.review_status,
                )
            )

        if filters.needs_review:
            stmt = stmt.where(
                or_(
                    feedback_exists(FILTER_FEEDBACK.c.rating == FeedbackRating.DOWN.value),
                    ChatLogModel.outcome.in_(TRIAGE_OUTCOMES),
                ),
                ~reviewed(),
            )

        return stmt

    async def list_reviews(self, log_ids: list[int]) -> list[ChatLogReview]:
        if not log_ids:
            return []

        rows = await self._db.execute(
            select(chat_log_reviews).where(chat_log_reviews.c.chat_log_id.in_(log_ids))
        )
        return [ChatLogReview(**row) for row in rows.mappings().all()]

    async def lock_review(self, log_id: int) -> ChatLogReview:
        await self._db.execute(
            insert(chat_log_reviews)
            .values(chat_log_id=log_id, status=ReviewStatus.NEW.value)
            .on_conflict_do_nothing(index_elements=["chat_log_id"])
        )
        row = await self._db.execute(
            select(chat_log_reviews).where(chat_log_reviews.c.chat_log_id == log_id).with_for_update()
        )
        return ChatLogReview(**row.mappings().one())

    async def save_review(self, review: ChatLogReview) -> None:
        values = {
            "status": review.status.value,
            "note": review.note,
            "reviewer_id": review.reviewer_id,
            "benchmark_question_id": review.benchmark_question_id,
            "update_date": review.update_date,
        }
        await self._db.execute(
            insert(chat_log_reviews)
            .values(chat_log_id=review.chat_log_id, **values)
            .on_conflict_do_update(index_elements=["chat_log_id"], set_=values)
        )

    async def stats(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> dict[str, Any]:
        def percentile(fraction: float, column: SQLColumnExpression[Any]) -> ColumnElement[Any]:
            return func.percentile_cont(fraction).within_group(column)

        down = func.count().filter(message_feedback.c.rating == FeedbackRating.DOWN.value)

        def scoped(*columns: SQLColumnExpression[Any]) -> Select[Any]:
            base = select(*columns).select_from(ChatLogModel).outerjoin(message_feedback, FEEDBACK_JOIN)
            return self.apply_filters(base, filters)

        totals = (
            (
                await self._db.execute(
                    scoped(
                        func.count().label("total"),
                        func.count()
                        .filter(ChatLogModel.outcome == AnswerOutcome.CACHE_HIT.value)
                        .label("cache_hits"),
                        func.count().filter(message_feedback.c.rating == FeedbackRating.UP.value).label("up"),
                        down.label("down"),
                        percentile(0.5, ChatLogModel.latency_ms).label("latency_p50"),
                        percentile(0.95, ChatLogModel.latency_ms).label("latency_p95"),
                        percentile(0.5, ChatLogModel.ttft_ms).label("ttft_p50"),
                        percentile(0.95, ChatLogModel.ttft_ms).label("ttft_p95"),
                        func.coalesce(func.sum(ChatLogModel.input_tokens), 0).label("input_tokens"),
                        func.coalesce(func.sum(ChatLogModel.output_tokens), 0).label("output_tokens"),
                        func.coalesce(func.sum(ChatLogModel.aux_tokens), 0).label("aux_tokens"),
                    )
                )
            )
            .mappings()
            .one()
        )

        async def counts(
            column: SQLColumnExpression[Any], *where: SQLColumnExpression[bool]
        ) -> dict[str, int]:
            stmt = scoped(column, func.count()).where(*where).group_by(column).order_by(func.count().desc())
            return {str(key): value for key, value in (await self._db.execute(stmt)).all()}

        reason = (
            func.jsonb_array_elements_text(message_feedback.c.reasons).table_valued("value").alias("reason")
        )
        reason_rows = await self._db.execute(
            scoped(reason.c.value, func.count())
            .join(reason, message_feedback.c.id.is_not(None))
            .group_by(reason.c.value)
            .order_by(func.count().desc())
        )

        async def groups(
            column: SQLColumnExpression[Any], limit: int | None = STATS_GROUP_LIMIT
        ) -> list[dict[str, Any]]:
            stmt = (
                scoped(
                    column.label("key"),
                    func.count().label("total"),
                    down.label("down"),
                    func.count()
                    .filter(ChatLogModel.outcome == AnswerOutcome.NOT_FOUND.value)
                    .label("not_found"),
                    func.count().filter(ChatLogModel.outcome == AnswerOutcome.ERROR.value).label("errors"),
                    func.count()
                    .filter(ChatLogModel.outcome == AnswerOutcome.CACHE_HIT.value)
                    .label("cache_hits"),
                    percentile(0.5, ChatLogModel.latency_ms).label("latency_p50"),
                    percentile(0.95, ChatLogModel.latency_ms).label("latency_p95"),
                )
                .group_by(column)
                .order_by(func.count().desc())
            )

            if limit is not None:
                stmt = stmt.limit(limit)

            return [dict(row) for row in (await self._db.execute(stmt)).mappings().all()]

        day = func.date_trunc("day", ChatLogModel.creation_date)
        by_day = sorted(await groups(day, limit=None), key=lambda row: row["key"])
        return {
            **dict(totals),
            "outcomes": await counts(ChatLogModel.outcome),
            "error_codes": await counts(ChatLogModel.error_code, ChatLogModel.error_code.is_not(None)),
            "reasons": {str(key): value for key, value in reason_rows.all()},
            "by_day": [{**row, "key": row["key"].date().isoformat()} for row in by_day],
            "by_space": await groups(ChatLogModel.space_id),
            "by_model": await groups(ChatLogModel.model_used),
            "by_config_hash": await groups(ChatLogModel.config_hash),
        }

    @staticmethod
    def to_entity(orm: ChatLogModel) -> ChatLog:
        data = {field.name: getattr(orm, field.name) for field in fields(ChatLog)}
        data["sources"] = data["sources"] or []
        data["outcome"] = AnswerOutcome(data["outcome"])
        return ChatLog(**data)
