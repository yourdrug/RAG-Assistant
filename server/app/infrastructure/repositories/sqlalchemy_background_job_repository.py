"""SQLAlchemy ORM implementation of BackgroundJobRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from domain.repositories.background_job_repository import BackgroundJob
from domain.value_objects.job_status import BackgroundJobStatus
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import BackgroundJobModel


class SQLAlchemyBackgroundJobRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    @staticmethod
    def _to_entity(orm: BackgroundJobModel) -> BackgroundJob:
        return BackgroundJob(
            id=orm.id,
            job_type=orm.job_type,
            status=orm.status,
            related_id=orm.related_id,
            request_id=orm.request_id,
            started_at=orm.started_at,
            finished_at=orm.finished_at,
            error_message=orm.error_message,
            creation_date=orm.creation_date,
        )

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
            orm.status = BackgroundJobStatus.RUNNING.value
            orm.started_at = datetime.now(tz=UTC)
            await self._db.flush()

    async def mark_done(self, job_id: int) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.status = BackgroundJobStatus.DONE.value
            orm.finished_at = datetime.now(tz=UTC)
            await self._db.flush()

    async def mark_failed(self, job_id: int, error: str) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.status = BackgroundJobStatus.FAILED.value
            orm.finished_at = datetime.now(tz=UTC)
            orm.error_message = error
            await self._db.flush()

    async def count_active(self) -> int:
        result = await self._db.execute(
            select(func.count())
            .select_from(BackgroundJobModel)
            .where(
                BackgroundJobModel.status.in_(
                    [
                        BackgroundJobStatus.PENDING.value,
                        BackgroundJobStatus.RUNNING.value,
                    ]
                )
            )
        )
        return result.scalar_one() or 0

    async def delete_old(self, days: int) -> int:
        cutoff = datetime.now(tz=UTC) - timedelta(days=days)
        result = await self._db.execute(
            delete(BackgroundJobModel).where(BackgroundJobModel.creation_date < cutoff)
        )
        await self._db.flush()
        return result.rowcount  # type: ignore[return-value]

    async def list_recent(self, limit: int = 50, offset: int = 0) -> list[BackgroundJob]:
        result = await self._db.execute(
            select(BackgroundJobModel)
            .order_by(BackgroundJobModel.creation_date.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = result.scalars().all()
        return [self._to_entity(orm) for orm in rows]

    async def get_by_id(self, job_id: int) -> BackgroundJob | None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()
        if not orm:
            return None
        return self._to_entity(orm)

    async def count_by_status(self) -> dict[str, int]:
        result = await self._db.execute(
            select(BackgroundJobModel.status, func.count()).group_by(BackgroundJobModel.status)
        )
        return {row[0]: row[1] for row in result.all()}

    async def recover_orphaned(self, timeout_minutes: int = 15) -> list[int]:
        result = await self._db.execute(
            text(
                """
                UPDATE background_jobs
                SET status = 'failed',
                    error_message = 'Worker died or restarted — task orphaned',
                    finished_at = NOW()
                WHERE status = 'running'
                  AND started_at < NOW() - make_interval(mins => :timeout)
                RETURNING id
                """
            ),
            {"timeout": timeout_minutes},
        )
        return [row[0] for row in result.fetchall()]
