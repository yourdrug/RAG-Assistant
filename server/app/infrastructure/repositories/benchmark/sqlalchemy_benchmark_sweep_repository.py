"""SQLAlchemy implementation of BenchmarkSweepRepository."""

from __future__ import annotations

from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.exceptions import BusinessRuleViolation
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import BENCHMARK_ACTIVE_SWEEP_INDEX, BenchmarkSweepModel


class SQLAlchemyBenchmarkSweepRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, sweep_id: int) -> BenchmarkSweep | None:
        stmt = select(BenchmarkSweepModel).where(BenchmarkSweepModel.id == sweep_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self.to_entity(orm) if orm else None

    async def create(self, sweep: BenchmarkSweep) -> BenchmarkSweep:
        orm = BenchmarkSweepModel(
            strategy=sweep.strategy,
            search_space=sweep.search_space,
            objective_weights=sweep.objective_weights,
            dataset=sweep.dataset,
            top_n_llm=sweep.top_n_llm,
            judge_model=sweep.judge_model,
            evaluation_mode=sweep.evaluation_mode,
            status=sweep.status,
            job_id=sweep.job_id,
            total_configs=sweep.total_configs,
            evaluated_configs=sweep.evaluated_configs,
            best_run_id=sweep.best_run_id,
        )
        self._db.add(orm)
        try:
            await self._db.flush()
        except IntegrityError as exc:
            # asyncpg exposes the constraint on the wrapped cause; psycopg
            # exposes it on diag. Other integrity failures must propagate.
            cause = exc.orig.__cause__ if exc.orig is not None else None
            diag = getattr(exc.orig, "diag", None)
            constraint = getattr(cause, "constraint_name", None) or getattr(diag, "constraint_name", None)
            if constraint == BENCHMARK_ACTIVE_SWEEP_INDEX:
                raise BusinessRuleViolation(
                    "Another sweep is pending or running — cancel it or wait for it to finish"
                ) from exc
            raise
        await self._db.refresh(orm)
        return self.to_entity(orm)

    async def update_status(self, sweep_id: int, status: str) -> None:
        stmt = (
            update(BenchmarkSweepModel)
            .where(BenchmarkSweepModel.id == sweep_id)
            .values(status=status, version=BenchmarkSweepModel.version + 1)
        )
        await self._db.execute(stmt)

    async def increment_evaluated(self, sweep_id: int) -> None:
        """Atomically increment evaluated_configs (SQL-level, no read-modify-write race)."""
        stmt = (
            update(BenchmarkSweepModel)
            .where(BenchmarkSweepModel.id == sweep_id)
            .values(
                evaluated_configs=BenchmarkSweepModel.evaluated_configs + 1,
                version=BenchmarkSweepModel.version + 1,
            )
        )
        await self._db.execute(stmt)

    async def set_best_run(self, sweep_id: int, run_id: int) -> None:
        stmt = (
            update(BenchmarkSweepModel)
            .where(BenchmarkSweepModel.id == sweep_id)
            .values(best_run_id=run_id, version=BenchmarkSweepModel.version + 1)
        )
        await self._db.execute(stmt)

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
        return [self.to_entity(orm) for orm in result.scalars().all()]

    async def count(self) -> int:
        stmt = select(func.count()).select_from(BenchmarkSweepModel)
        result = await self._db.execute(stmt)
        return result.scalar_one()

    async def has_active(self) -> bool:
        stmt = (
            select(func.count())
            .select_from(BenchmarkSweepModel)
            .where(
                BenchmarkSweepModel.status.in_(
                    [
                        BenchmarkSweepStatus.PENDING.value,
                        BenchmarkSweepStatus.RUNNING.value,
                    ]
                )
            )
        )
        result = await self._db.execute(stmt)
        return (result.scalar_one() or 0) > 0

    @staticmethod
    def to_entity(orm: BenchmarkSweepModel) -> BenchmarkSweep:
        return BenchmarkSweep(
            id=orm.id,
            creation_date=orm.creation_date,
            status=orm.status,
            strategy=orm.strategy,
            search_space=orm.search_space or {},
            objective_weights=orm.objective_weights or {},
            dataset=orm.dataset,
            top_n_llm=orm.top_n_llm,
            judge_model=orm.judge_model,
            evaluation_mode=orm.evaluation_mode,
            job_id=orm.job_id,
            total_configs=orm.total_configs,
            evaluated_configs=orm.evaluated_configs,
            best_run_id=orm.best_run_id,
            version=orm.version,
        )
