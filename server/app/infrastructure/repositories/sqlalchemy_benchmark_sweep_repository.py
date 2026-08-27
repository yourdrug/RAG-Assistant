"""SQLAlchemy implementation of BenchmarkSweepRepository."""

from __future__ import annotations

from domain.entities.benchmark_sweep import BenchmarkSweep
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import BenchmarkSweepModel


class SQLAlchemyBenchmarkSweepRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, sweep_id: int) -> BenchmarkSweep | None:
        stmt = select(BenchmarkSweepModel).where(BenchmarkSweepModel.id == sweep_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def create(self, sweep: BenchmarkSweep) -> BenchmarkSweep:
        orm = BenchmarkSweepModel(
            strategy=sweep.strategy,
            search_space=sweep.search_space,
            objective_weights=sweep.objective_weights,
            dataset=sweep.dataset,
            top_n_llm=sweep.top_n_llm,
            status=sweep.status,
            job_id=sweep.job_id,
            total_configs=sweep.total_configs,
            evaluated_configs=sweep.evaluated_configs,
            best_run_id=sweep.best_run_id,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        return self._to_entity(orm)

    async def update_status(self, sweep_id: int, status: str) -> None:
        stmt = select(BenchmarkSweepModel).where(BenchmarkSweepModel.id == sweep_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm:
            orm.status = status
            await self._db.flush()

    async def increment_evaluated(self, sweep_id: int) -> None:
        stmt = select(BenchmarkSweepModel).where(BenchmarkSweepModel.id == sweep_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm:
            orm.evaluated_configs += 1
            await self._db.flush()

    async def set_best_run(self, sweep_id: int, run_id: int) -> None:
        stmt = select(BenchmarkSweepModel).where(BenchmarkSweepModel.id == sweep_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm:
            orm.best_run_id = run_id
            await self._db.flush()

    async def list_items(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkSweep]:
        stmt = (
            select(BenchmarkSweepModel)
            .order_by(BenchmarkSweepModel.creation_date.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def count(self) -> int:
        stmt = select(func.count()).select_from(BenchmarkSweepModel)
        result = await self._db.execute(stmt)
        return result.scalar_one()

    @staticmethod
    def _to_entity(orm: BenchmarkSweepModel) -> BenchmarkSweep:
        return BenchmarkSweep(
            id=orm.id,
            creation_date=orm.creation_date,
            status=orm.status,
            strategy=orm.strategy,
            search_space=orm.search_space or {},
            objective_weights=orm.objective_weights or {},
            dataset=orm.dataset,
            top_n_llm=orm.top_n_llm,
            job_id=orm.job_id,
            total_configs=orm.total_configs,
            evaluated_configs=orm.evaluated_configs,
            best_run_id=orm.best_run_id,
        )
