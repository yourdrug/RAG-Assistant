"""Real PostgreSQL contracts for optional benchmark annotations."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.benchmark.question import BenchmarkQuestion
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_annotations_crud_round_trip(pg_engine: AsyncEngine) -> None:
    annotations = {
        "expected_fragments": [{"source": "rules.pdf", "text": "Only organizations"}],
        "required_conditions": ["Only organizations"],
        "expected_refusal": False,
    }
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkQuestionRepository(session)
        created = await repo.create(
            BenchmarkQuestion(
                question="Q",
                annotations=annotations,
                history=[{"role": "user", "content": "Earlier question"}],
                as_of_date=date(2020, 1, 1),
            )
        )
        assert created.id is not None
        assert created.annotations == annotations
        assert (
            await repo.bulk_create(
                [
                    BenchmarkQuestion(question="Legacy Q"),
                    BenchmarkQuestion(question="New Q", annotations=annotations),
                ]
            )
            == 2
        )
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkQuestionRepository(session)
        restored = await repo.get_by_id(created.id)
        assert restored is not None
        assert restored.annotations == annotations
        assert restored.history == [{"role": "user", "content": "Earlier question"}]
        assert restored.as_of_date == date(2020, 1, 1)
        updated = await repo.update(created.id, history=None, as_of_date=None)
        assert updated is not None
        assert updated.history is None
        assert len(await repo.list_items()) == 3
        updated = await repo.update(created.id, annotations=None)
        assert updated is not None
        assert updated.annotations is None


async def test_annotations_default_and_object_constraint(pg_engine: AsyncEngine) -> None:

    async with pg_engine.begin() as conn:
        question_id = (
            await conn.execute(
                text(
                    "INSERT INTO benchmark_questions (question, creation_date) "
                    "VALUES ('Legacy question', CURRENT_TIMESTAMP) RETURNING id"
                )
            )
        ).scalar_one()
        assert (
            await conn.execute(
                text("SELECT question, annotations FROM benchmark_questions WHERE id=:id"),
                {"id": question_id},
            )
        ).one() == ("Legacy question", None)
        with pytest.raises(IntegrityError):
            async with conn.begin_nested():
                await conn.execute(
                    text("UPDATE benchmark_questions SET annotations='[]'::jsonb WHERE id=:id"),
                    {"id": question_id},
                )
        await conn.execute(
            text("UPDATE benchmark_questions SET annotations='{}'::jsonb WHERE id=:id"), {"id": question_id}
        )
        assert (
            await conn.execute(
                text("SELECT question FROM benchmark_questions WHERE id=:id"), {"id": question_id}
            )
        ).scalar_one() == "Legacy question"
