"""SQLAlchemy implementation of BenchmarkQuestionRepository."""

from __future__ import annotations

from domain.entities.benchmark_question import BenchmarkQuestion
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import BenchmarkQuestionModel


class SQLAlchemyBenchmarkQuestionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, question_id: int) -> BenchmarkQuestion | None:
        stmt = select(BenchmarkQuestionModel).where(BenchmarkQuestionModel.id == question_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def list(
        self,
        *,
        dataset: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkQuestion]:
        stmt = select(BenchmarkQuestionModel).order_by(BenchmarkQuestionModel.creation_date.desc())
        stmt = self._apply_filters(stmt, dataset=dataset, tag=tag, search=search, is_active=is_active)
        stmt = stmt.offset(offset).limit(limit)
        result = await self._db.execute(stmt)
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def count(
        self,
        *,
        dataset: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(BenchmarkQuestionModel)
        stmt = self._apply_filters(stmt, dataset=dataset, tag=tag, search=search, is_active=is_active)
        result = await self._db.execute(stmt)
        return result.scalar_one()

    async def create(self, question: BenchmarkQuestion) -> BenchmarkQuestion:
        orm = BenchmarkQuestionModel(
            question=question.question,
            expected_answer=question.expected_answer,
            source_hint=question.source_hint,
            tags=question.tags,
            dataset=question.dataset,
            is_active=question.is_active,
            created_by=question.created_by,
            notes=question.notes,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        return self._to_entity(orm)

    async def update(self, question_id: int, **fields) -> BenchmarkQuestion | None:
        stmt = select(BenchmarkQuestionModel).where(BenchmarkQuestionModel.id == question_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        for key, value in fields.items():
            if hasattr(orm, key):
                setattr(orm, key, value)
        await self._db.flush()
        await self._db.refresh(orm)
        return self._to_entity(orm)

    async def delete(self, question_id: int) -> bool:
        stmt = select(BenchmarkQuestionModel).where(BenchmarkQuestionModel.id == question_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            return False
        await self._db.delete(orm)
        return True

    async def bulk_create(self, questions: list[BenchmarkQuestion]) -> int:
        orms = [
            BenchmarkQuestionModel(
                question=q.question,
                expected_answer=q.expected_answer,
                source_hint=q.source_hint,
                tags=q.tags,
                dataset=q.dataset,
                is_active=q.is_active,
                created_by=q.created_by,
                notes=q.notes,
            )
            for q in questions
        ]
        self._db.add_all(orms)
        await self._db.flush()
        return len(orms)

    async def get_datasets(self) -> list[str]:
        stmt = select(BenchmarkQuestionModel.dataset).distinct().order_by(BenchmarkQuestionModel.dataset)
        result = await self._db.execute(stmt)
        return [row[0] for row in result.all()]

    async def count_by_dataset(self) -> dict[str, int]:
        stmt = select(
            BenchmarkQuestionModel.dataset,
            func.count(),
        ).group_by(BenchmarkQuestionModel.dataset)
        result = await self._db.execute(stmt)
        return {row[0]: row[1] for row in result.all()}

    @staticmethod
    def _apply_filters(stmt, *, dataset=None, tag=None, search=None, is_active=None):
        if dataset is not None:
            stmt = stmt.where(BenchmarkQuestionModel.dataset == dataset)
        if tag is not None:
            stmt = stmt.where(BenchmarkQuestionModel.tags.op("@>")(f'["{tag}"]'))
        if search:
            pattern = f"%{search}%"
            stmt = stmt.where(
                BenchmarkQuestionModel.question.ilike(pattern)
                | BenchmarkQuestionModel.expected_answer.ilike(pattern)
                | BenchmarkQuestionModel.notes.ilike(pattern)
            )
        if is_active is not None:
            stmt = stmt.where(BenchmarkQuestionModel.is_active == is_active)
        return stmt

    @staticmethod
    def _to_entity(orm: BenchmarkQuestionModel) -> BenchmarkQuestion:
        return BenchmarkQuestion(
            id=orm.id,
            creation_date=orm.creation_date,
            question=orm.question,
            expected_answer=orm.expected_answer,
            source_hint=orm.source_hint,
            tags=orm.tags or [],
            dataset=orm.dataset,
            is_active=orm.is_active,
            created_by=orm.created_by,
            notes=orm.notes,
        )
