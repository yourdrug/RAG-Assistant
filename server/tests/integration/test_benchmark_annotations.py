"""Real PostgreSQL contracts for optional benchmark annotations."""

from importlib import import_module

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.entities.benchmark_question import BenchmarkQuestion
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


async def test_annotation_migration_preserves_questions_and_rejects_arrays(pg_engine):
    migration = import_module(
        "infrastructure.database.migrations.versions.g0b1c2d3e4f5_add_benchmark_annotations"
    )

    def migrate(conn, operation):
        with Operations.context(MigrationContext.configure(conn)):
            operation()

    async with pg_engine.begin() as conn:
        await conn.run_sync(lambda c: migrate(c, migration.downgrade))
        question_id = (
            await conn.execute(
                text(
                    "INSERT INTO benchmark_questions (question, creation_date) "
                    "VALUES ('Legacy question', CURRENT_TIMESTAMP) RETURNING id"
                )
            )
        ).scalar_one()
        await conn.run_sync(lambda c: migrate(c, migration.upgrade))
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
        await conn.run_sync(lambda c: migrate(c, migration.downgrade))
        assert (
            await conn.execute(
                text("SELECT question FROM benchmark_questions WHERE id=:id"), {"id": question_id}
            )
        ).scalar_one() == "Legacy question"
