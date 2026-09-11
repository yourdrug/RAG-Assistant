"""SQLAlchemy implementation of BenchmarkRunRepository."""

from __future__ import annotations

from domain.entities.benchmark_run import BenchmarkRun
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import BenchmarkRunModel


class SQLAlchemyBenchmarkRunRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, run_id: int) -> BenchmarkRun | None:
        stmt = select(BenchmarkRunModel).where(BenchmarkRunModel.id == run_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def create(self, run: BenchmarkRun) -> BenchmarkRun:
        orm = BenchmarkRunModel(
            sweep_id=run.sweep_id,
            config_json=run.config_json,
            summary_metrics=run.summary_metrics,
            duration_sec=run.duration_sec,
            llm_evaluated=run.llm_evaluated,
            dataset=run.dataset,
            per_question_results=run.per_question_results,
            filename=run.filename,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        return self._to_entity(orm)

    async def list_items(
        self,
        *,
        sweep_id: int | None = None,
        dataset: str | None = None,
        sort_by: str = "creation_date",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkRun]:
        sort_col = getattr(BenchmarkRunModel, sort_by, BenchmarkRunModel.creation_date)
        if sort_order == "asc":
            stmt = select(BenchmarkRunModel).order_by(sort_col.asc())
        else:
            stmt = select(BenchmarkRunModel).order_by(sort_col.desc())

        if sweep_id is not None:
            stmt = stmt.where(BenchmarkRunModel.sweep_id == sweep_id)
        if dataset is not None:
            stmt = stmt.where(BenchmarkRunModel.dataset == dataset)

        stmt = stmt.offset(offset).limit(limit)
        result = await self._db.execute(stmt)
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def count(
        self,
        *,
        sweep_id: int | None = None,
        dataset: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(BenchmarkRunModel)
        if sweep_id is not None:
            stmt = stmt.where(BenchmarkRunModel.sweep_id == sweep_id)
        if dataset is not None:
            stmt = stmt.where(BenchmarkRunModel.dataset == dataset)
        result = await self._db.execute(stmt)
        return result.scalar_one()

    async def get_by_ids(self, ids: list[int]) -> list[BenchmarkRun]:
        if not ids:
            return []
        stmt = (
            select(BenchmarkRunModel)
            .where(BenchmarkRunModel.id.in_(ids))
            .order_by(BenchmarkRunModel.creation_date.desc())
        )
        result = await self._db.execute(stmt)
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def get_latest(self, dataset: str | None = None) -> BenchmarkRun | None:
        stmt = select(BenchmarkRunModel).order_by(BenchmarkRunModel.creation_date.desc())
        if dataset is not None:
            stmt = stmt.where(BenchmarkRunModel.dataset == dataset)
        stmt = stmt.limit(1)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    @staticmethod
    def _to_entity(orm: BenchmarkRunModel) -> BenchmarkRun:
        return BenchmarkRun(
            id=orm.id,
            creation_date=orm.creation_date,
            sweep_id=orm.sweep_id,
            config_json=orm.config_json or {},
            summary_metrics=orm.summary_metrics or {},
            duration_sec=orm.duration_sec,
            llm_evaluated=orm.llm_evaluated,
            dataset=orm.dataset,
            per_question_results=orm.per_question_results,
            filename=orm.filename,
        )
