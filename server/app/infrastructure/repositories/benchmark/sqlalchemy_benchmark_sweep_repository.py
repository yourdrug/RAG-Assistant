"""SQLAlchemy implementation of BenchmarkSweepRepository."""

from __future__ import annotations

from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.exceptions import BusinessRuleViolation
from domain.value_objects.job_status import BackgroundJobStatus
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from infrastructure.database.models import (
    BENCHMARK_ACTIVE_SWEEP_INDEX,
    BackgroundJobModel,
    BenchmarkSweepModel,
    BenchmarkCheckpointModel,
)


class SQLAlchemyBenchmarkSweepRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, sweep_id: int, *, for_update: bool = False) -> BenchmarkSweep | None:
        stmt = select(BenchmarkSweepModel).where(BenchmarkSweepModel.id == sweep_id)
        if for_update:
            stmt = stmt.with_for_update()
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

    async def requeue(self, sweep_id: int) -> bool:
        # Compare-and-set plus the unique active-sweep index serialize resumes.
        try:
            result = await self._db.execute(
                update(BenchmarkSweepModel)
                .where(
                    BenchmarkSweepModel.id == sweep_id,
                    BenchmarkSweepModel.status.in_(
                        [BenchmarkSweepStatus.FAILED.value, BenchmarkSweepStatus.CANCELLED.value]
                    ),
                )
                .values(
                    status=BenchmarkSweepStatus.PENDING.value,
                    job_id=None,
                    version=BenchmarkSweepModel.version + 1,
                )
            )
        except IntegrityError as exc:
            cause = exc.orig.__cause__ if exc.orig is not None else None
            diag = getattr(exc.orig, "diag", None)
            constraint = getattr(cause, "constraint_name", None) or getattr(diag, "constraint_name", None)
            if constraint == BENCHMARK_ACTIVE_SWEEP_INDEX:
                raise BusinessRuleViolation("Another sweep is already active") from exc
            raise
        return result.rowcount == 1

    async def fail_dead_jobs(self) -> int:
        result = await self._db.execute(
            update(BenchmarkSweepModel)
            .where(
                BenchmarkSweepModel.status.in_(
                    [BenchmarkSweepStatus.PENDING.value, BenchmarkSweepStatus.RUNNING.value]
                ),
                BenchmarkSweepModel.job_id.in_(
                    select(BackgroundJobModel.id).where(
                        BackgroundJobModel.status == BackgroundJobStatus.FAILED.value
                    )
                ),
            )
            .values(status=BenchmarkSweepStatus.FAILED.value, version=BenchmarkSweepModel.version + 1)
        )
        return result.rowcount

    async def set_job_id(self, sweep_id: int, job_id: int) -> None:
        await self._db.execute(
            update(BenchmarkSweepModel)
            .where(BenchmarkSweepModel.id == sweep_id)
            .values(job_id=job_id, version=BenchmarkSweepModel.version + 1)
        )

    async def update_progress(self, sweep_id: int, evaluated: int, total: int) -> None:
        await self._db.execute(
            update(BenchmarkSweepModel)
            .where(BenchmarkSweepModel.id == sweep_id)
            .values(
                evaluated_configs=evaluated,
                total_configs=total,
                version=BenchmarkSweepModel.version + 1,
            )
        )

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

    async def load_checkpoint(self, sweep_id: int, key: str) -> object | None:
        row = await self._db.get(BenchmarkCheckpointModel, (sweep_id, key))
        return row.payload["value"] if row is not None else None

    async def save_checkpoint(self, sweep_id: int, key: str, value: object) -> None:
        statement = insert(BenchmarkCheckpointModel).values(
            sweep_id=sweep_id, key=key, payload={"value": value}
        )
        await self._db.execute(
            statement.on_conflict_do_update(
                index_elements=[BenchmarkCheckpointModel.sweep_id, BenchmarkCheckpointModel.key],
                set_={"payload": statement.excluded.payload},
            )
        )
