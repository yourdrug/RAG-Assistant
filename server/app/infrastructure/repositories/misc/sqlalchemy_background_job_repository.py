"""SQLAlchemy ORM implementation of BackgroundJobRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select, text

from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus
from infrastructure.database.models import BackgroundJobModel


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SQLAlchemyBackgroundJobRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    @staticmethod
    def _to_entity(orm: BackgroundJobModel) -> BackgroundJob:
        return BackgroundJob(
            id=orm.id,
            job_type=orm.job_type,
            status=BackgroundJobStatus(orm.status),
            related_id=orm.related_id,
            request_id=orm.request_id,
            started_at=orm.started_at,
            finished_at=orm.finished_at,
            error_message=orm.error_message,
            creation_date=orm.creation_date,
            dispatch_payload=orm.dispatch_payload,
        )

    async def create(self, job: BackgroundJob) -> BackgroundJob:
        orm = BackgroundJobModel(
            job_type=job.job_type,
            status=job.status,
            related_id=job.related_id,
            request_id=job.request_id,
            dispatch_payload=job.dispatch_payload,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        job.id = orm.id
        job.creation_date = orm.creation_date
        return job

    async def pending_dispatch(self, limit: int = 100) -> list[BackgroundJob]:
        """Lock pending durable submissions until the caller commits or rolls back."""
        result = await self._db.execute(
            select(BackgroundJobModel)
            .where(
                BackgroundJobModel.status == BackgroundJobStatus.PENDING.value,
                BackgroundJobModel.dispatch_payload.is_not(None),
            )
            .order_by(BackgroundJobModel.heartbeat_at.asc().nulls_first(), BackgroundJobModel.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def mark_running(self, job_id: int) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()

        if orm:
            entity = self._to_entity(orm)
            entity.mark_running()
            orm.status = entity.status
            orm.started_at = entity.started_at
            await self._db.flush()

    async def mark_done(self, job_id: int) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()

        if orm:
            entity = self._to_entity(orm)
            entity.mark_done()
            orm.status = entity.status
            orm.finished_at = entity.finished_at
            await self._db.flush()

    async def mark_failed(self, job_id: int, error: str) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()

        if orm:
            entity = self._to_entity(orm)
            entity.mark_failed(error)
            orm.status = entity.status
            orm.finished_at = entity.finished_at
            orm.error_message = entity.error_message
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
            delete(BackgroundJobModel).where(
                BackgroundJobModel.creation_date < cutoff,
                BackgroundJobModel.status.in_(
                    [BackgroundJobStatus.DONE.value, BackgroundJobStatus.FAILED.value]
                ),
            )
        )
        await self._db.flush()
        return result.rowcount or 0

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

    async def touch_heartbeat(self, job_id: int) -> None:
        result = await self._db.execute(select(BackgroundJobModel).where(BackgroundJobModel.id == job_id))
        orm = result.scalar_one_or_none()

        if orm:
            orm.heartbeat_at = datetime.now(tz=UTC)
            await self._db.flush()

    async def fail_stale_pending(self, timeout_minutes: int) -> list[BackgroundJob]:
        cutoff = datetime.now(tz=UTC) - timedelta(minutes=timeout_minutes)
        result = await self._db.execute(
            select(BackgroundJobModel).where(
                BackgroundJobModel.status == BackgroundJobStatus.PENDING.value,
                func.coalesce(BackgroundJobModel.heartbeat_at, BackgroundJobModel.creation_date) < cutoff,
                BackgroundJobModel.dispatch_payload.is_(None),
            )
        )
        orms = result.scalars().all()
        failed = []

        for orm in orms:
            entity = self._to_entity(orm)
            entity.mark_failed("Stale pending — exceeded timeout")
            orm.status = entity.status
            orm.finished_at = entity.finished_at
            orm.error_message = entity.error_message
            failed.append(entity)

        await self._db.flush()
        return failed
