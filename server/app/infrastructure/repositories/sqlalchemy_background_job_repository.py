"""SQLAlchemy ORM implementation of BackgroundJobRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from domain.repositories.background_job_repository import BackgroundJob
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import BackgroundJobModel


class SQLAlchemyBackgroundJobRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, job: BackgroundJob) -> BackgroundJob:
        orm = BackgroundJobModel(
            job_type=job.job_type,
            status=job.status,
            related_id=job.related_id,
            request_id=job.request_id,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        job.id = orm.id
        job.creation_date = orm.creation_date
        return job

    async def mark_running(self, job_id: int) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.status = "running"
            orm.started_at = datetime.now(tz=UTC)
            await self._db.flush()

    async def mark_done(self, job_id: int) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.status = "done"
            orm.finished_at = datetime.now(tz=UTC)
            await self._db.flush()

    async def mark_failed(self, job_id: int, error: str) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.status = "failed"
            orm.finished_at = datetime.now(tz=UTC)
            orm.error_message = error
            await self._db.flush()

    async def count_active(self) -> int:
        result = await self._db.execute(
            select(func.count())
            .select_from(BackgroundJobModel)
            .where(BackgroundJobModel.status.in_(["pending", "running"]))
        )
        return result.scalar_one() or 0

    async def delete_old(self, days: int) -> int:
        cutoff = datetime.now(tz=UTC) - timedelta(days=days)
        result = await self._db.execute(
            delete(BackgroundJobModel).where(BackgroundJobModel.creation_date < cutoff)
        )
        await self._db.flush()
        return result.rowcount  # type: ignore[return-value]
