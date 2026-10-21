"""Real PostgreSQL contracts for optional benchmark annotations."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.benchmark.question import BenchmarkQuestion
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_annotations_crud_round_trip(pg_engine):
    annotations = {
        "expected_fragments": [{"source": "rules.pdf", "text": "Only organizations"}],
        "required_conditions": ["Only organizations"],
        "expected_refusal": False,
    }
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBenchmarkQuestionRepository(session)
        created = await repo.create(BenchmarkQuestion(question="Q", annotations=annotations))
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
        assert (await repo.get_by_id(created.id)).annotations == annotations
        assert len(await repo.list_items()) == 3
        assert (await repo.update(created.id, annotations=None)).annotations is None


async def test_annotations_default_and_object_constraint(pg_engine):

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
