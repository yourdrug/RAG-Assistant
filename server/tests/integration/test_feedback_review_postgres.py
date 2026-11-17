"""Reactions and reviews in PostgreSQL: concurrent upserts and transfers, filters and stats."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.integration.uow import SessionDatabase

from application.chat import ChatLogService, MessageFeedbackService
from application.chat.review import BenchmarkTransfer
from domain.chat.chat_log import ChatLog
from domain.chat.feedback import FeedbackRating, FeedbackReason
from domain.chat.repositories import ChatLogFilter
from domain.chat.review import ReviewStatus
from domain.rag.outcome import AnswerOutcome
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.database.models import (
    BenchmarkQuestionModel,
    ConversationModel,
    MessageModel,
    UserModel,
    message_feedback,
)
from infrastructure.uow_factory import UnitOfWorkFactory


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


pytestmark = pytest.mark.integration


class Redactor:
    def redact(self, text: str) -> str:
        return text

    def scan_and_redact(self, text: str) -> tuple[str, list[str]]:
        return text, []


async def seed(engine: AsyncEngine) -> dict[str, int]:
    """Two users; three answered attempts of the owner and one of the other user."""
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    ids: dict[str, int] = {}
    async with sessions.begin() as session:
        for name in ("owner", "other"):
            user = UserModel(
                email=f"{name}@example.test",
                hashed_password="unused",  # noqa: S106 - disposable test user
                role="user",
                kind="internal",
            )
            session.add(user)
            await session.flush()
            conversation = ConversationModel(user_id=user.id, space_id=1)
            session.add(conversation)
            await session.flush()
            ids[name] = user.id
            ids[f"{name}_conversation"] = conversation.id

        for index in range(3):
            session.add(
                MessageModel(conversation_id=ids["owner_conversation"], role="user", content=f"q{index}")
            )
            answer = MessageModel(
                conversation_id=ids["owner_conversation"], role="assistant", content=f"a{index}"
            )
            session.add(answer)
            await session.flush()
            ids[f"answer{index}"] = answer.id

        answer = MessageModel(conversation_id=ids["other_conversation"], role="assistant", content="other")
        session.add(answer)
        await session.flush()
        ids["other_answer"] = answer.id

    return ids


async def journal(factory: UnitOfWorkFactory, ids: dict[str, int]) -> list[int]:
    entries = [
        ChatLog(
            user_id=ids["owner"],
            conversation_id=ids["owner_conversation"],
            message_id=ids["answer0"],
            question="q0",
            space_id=1,
            pack_key="generic",
            latency_ms=100,
            ttft_ms=10,
            model_used="model-a",
        ),
        ChatLog(
            user_id=ids["owner"],
            conversation_id=ids["owner_conversation"],
            message_id=ids["answer1"],
            question="q1",
            space_id=1,
            pack_key="generic",
            latency_ms=300,
            model_used="model-a",
            outcome=AnswerOutcome.CACHE_HIT,
        ),
        ChatLog(
            user_id=ids["owner"],
            conversation_id=ids["owner_conversation"],
            question="q-error",
            space_id=1,
            pack_key="generic",
            outcome=AnswerOutcome.ERROR,
            error_code="llm_unavailable",
            model_used="model-b",
        ),
        ChatLog(
            user_id=ids["other"],
            conversation_id=ids["other_conversation"],
            message_id=ids["other_answer"],
            question="other",
            space_id=1,
            pack_key="generic",
            outcome=AnswerOutcome.NOT_FOUND,
        ),
    ]
    async with factory.create(master=True) as uow:
        for entry in entries:
            await uow.chat_logs.save(entry)

    return [entry.id for entry in entries if entry.id is not None]


@pytest.mark.asyncio
async def test_concurrent_ratings_keep_one_row_and_db_enforces_reason_rule(pg_engine: AsyncEngine) -> None:
    ids = await seed(pg_engine)
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    service = MessageFeedbackService(UnitOfWorkFactory(SessionDatabase(sessions)), Redactor())

    results = await asyncio.gather(
        service.rate(ids["answer0"], ids["owner"], FeedbackRating.UP),
        service.rate(ids["answer0"], ids["owner"], FeedbackRating.DOWN, [FeedbackReason.WRONG_FACT], "c"),
    )

    async with sessions() as session:
        rows = (await session.execute(select(message_feedback))).mappings().all()

    assert len(rows) == 1
    assert rows[0]["rating"] in {result.rating for result in results}

    with pytest.raises(IntegrityError):
        async with sessions.begin() as session:
            await session.execute(
                text("UPDATE message_feedback SET rating = 'up', reasons = '[\"other\"]'::jsonb")
            )

    # Deleting the message removes its reaction.
    async with sessions.begin() as session:
        await session.execute(text("DELETE FROM messages WHERE id = :id"), {"id": ids["answer0"]})

    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(message_feedback)) == 0


@pytest.mark.asyncio
async def test_filters_stats_and_scope_follow_reactions_and_reviews(pg_engine: AsyncEngine) -> None:  # noqa: PLR0915 - one seeded journal, many filter checks
    ids = await seed(pg_engine)
    factory = UnitOfWorkFactory(SessionDatabase(async_sessionmaker(pg_engine, expire_on_commit=False)))
    ok, cached, error, foreign = await journal(factory, ids)
    feedback = MessageFeedbackService(factory, Redactor())
    await feedback.rate(
        ids["answer0"], ids["owner"], FeedbackRating.DOWN, [FeedbackReason.INCOMPLETE], "short"
    )
    await feedback.rate(ids["answer1"], ids["owner"], FeedbackRating.UP)
    service = ChatLogService(factory, pii_redactor=Redactor())

    async def found(**kwargs: Any) -> set[int]:
        return {row.entry.id for row in await service.list_rows(ChatLogFilter(**kwargs)) if row.entry.id}

    assert await found(rating="down") == {ok}
    assert await found(rating="up") == {cached}
    assert await found(rating="none") == {error, foreign}
    assert await found(reason="incomplete") == {ok}
    assert await found(has_comment=True) == {ok}
    assert await found(needs_review=True) == {ok, error, foreign}
    assert await found(needs_review=True, user_ids=(ids["owner"],)) == {ok, error}

    await service.reviews.set_status(error, ReviewStatus.PIPELINE_ISSUE, "retry", ids["other"], None)
    await service.reviews.set_status(foreign, ReviewStatus.NEW, None, ids["other"], None)
    assert await found(needs_review=True) == {ok, foreign}
    assert await found(review_status="new") == {ok, cached, foreign}
    assert await found(review_status="pipeline_issue") == {error}

    rows = await service.list_rows(ChatLogFilter(user_ids=(ids["owner"],)))
    by_id = {row.entry.id: row for row in rows}
    rated, reviewed = by_id[ok].feedback, by_id[error].review
    assert rated is not None
    assert rated.comment == "short"
    assert reviewed is not None
    assert reviewed.status == ReviewStatus.PIPELINE_ISSUE

    stats = await service.stats(ChatLogFilter(user_ids=(ids["owner"],)))
    assert stats["total"] == 3
    assert stats["up"] == 1
    assert stats["down"] == 1
    assert stats["cache_hits"] == 1
    assert stats["outcomes"] == {"generated": 1, "cache_hit": 1, "error": 1}
    assert stats["error_codes"] == {"llm_unavailable": 1}
    assert stats["reasons"] == {"incomplete": 1}
    assert stats["latency_p50"] == 200
    assert stats["ttft_p50"] == 10
    assert {row["key"]: row["total"] for row in stats["by_model"]} == {"model-a": 2, "model-b": 1}
    assert sum(row["total"] for row in stats["by_day"]) == 3
    assert stats["by_space"][0]["down"] == 1

    # Feedback filters add a subquery on message_feedback next to the stats outer join.
    cases: list[tuple[dict[str, Any], int, int]] = [
        ({"rating": "down"}, 1, 1),
        ({"reason": "incomplete"}, 1, 1),
        ({"has_comment": True}, 1, 1),
        ({"rating": "none"}, 2, 0),
        ({"needs_review": True}, 2, 1),
    ]

    for feedback_filter, total, down in cases:
        filtered = await service.stats(ChatLogFilter(**feedback_filter))
        assert (filtered["total"], filtered["down"]) == (total, down), feedback_filter

    assert (await service.stats(ChatLogFilter(reason="incomplete")))["reasons"] == {"incomplete": 1}

    exported = await service.export_rows(ChatLogFilter(), max_rows=2)
    assert len(exported) == 2


@pytest.mark.asyncio
async def test_concurrent_transfers_create_one_question(pg_engine: AsyncEngine) -> None:
    ids = await seed(pg_engine)
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    factory = UnitOfWorkFactory(SessionDatabase(sessions))
    ok, *_ = await journal(factory, ids)
    service = ChatLogService(factory, pii_redactor=Redactor())
    transfer = BenchmarkTransfer(dataset="generic-main", question="q0")

    results = await asyncio.gather(
        service.reviews.transfer(ok, transfer, ids["other"], None),
        service.reviews.transfer(ok, transfer, ids["other"], None),
        return_exceptions=True,
    )

    assert sum(isinstance(result, BusinessRuleViolationError) for result in results) == 1
    async with sessions() as session:
        questions = (await session.execute(select(BenchmarkQuestionModel))).scalars().all()

    assert [(question.dataset, question.is_active) for question in questions] == [("generic-main", False)]
    [row] = [row for row in await service.list_rows(ChatLogFilter()) if row.entry.id == ok]
    assert row.review is not None
    assert row.review.status == ReviewStatus.IN_BENCHMARK
    assert row.review.benchmark_question_id == questions[0].id
