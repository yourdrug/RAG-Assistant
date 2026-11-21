"""SQLAlchemy ORM implementation of BackgroundJobRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, or_, select, true, update
from sqlalchemy.exc import IntegrityError

from application.jobs.execution import ExecutionLeaseLostError
from config import JobsSection, process_section
from domain.runtime.jobs import (
    BackgroundJob,
    BackgroundJobStatus,
    IndexingQueueSnapshot,
    JobAdmissionError,
    JobConcurrencyLimitReachedError,
    JobType,
)
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.database.models import BackgroundJobModel
from infrastructure.metrics.jobs import measure_fencing
from infrastructure.telemetry.tracing import current_trace_carrier


if TYPE_CHECKING:
    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

    from domain.runtime.jobs import ConcurrencyLimit

# A job still sitting in the broker is re-published at most this often; the
# broker refuses the duplicate ID, and a lost delivery is restored within it.
REDISPATCH_AFTER = timedelta(seconds=60)


def expired_execution(now: datetime, timeout_minutes: int) -> ColumnElement[bool]:
    return or_(
        BackgroundJobModel.lease_until <= now,
        BackgroundJobModel.lease_until.is_(None)
        & (
            func.coalesce(BackgroundJobModel.heartbeat_at, BackgroundJobModel.started_at)
            < now - timedelta(minutes=timeout_minutes)
        ),
    )


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
            heartbeat_at=orm.heartbeat_at,
            payload_version=orm.payload_version,
            deduplication_key=orm.deduplication_key,
            attempt=orm.attempt,
            attempt_id=orm.attempt_id,
            lease_until=orm.lease_until,
            next_attempt_at=orm.next_attempt_at,
            dispatch_attempts=orm.dispatch_attempts,
            progress=orm.progress,
        )

    async def trace_context(self, job_id: int) -> dict[str, str] | None:
        """W3C trace context recorded when the job was submitted (None for untraced jobs)."""
        result = await self._db.execute(
            select(BackgroundJobModel.trace_context).where(BackgroundJobModel.id == job_id)
        )
        return result.scalar_one_or_none()

    async def create(self, job: BackgroundJob) -> BackgroundJob:
        if job.job_type in {
            JobType.DOCUMENT_PROCESSING,
            JobType.INGEST,
            JobType.INGEST_FILE,
            JobType.SYNC_RUN,
        }:
            await self.check_admission((job.dispatch_payload or {}).get("cost_principal"))

        orm = BackgroundJobModel(
            job_type=job.job_type,
            status=job.status,
            related_id=job.related_id,
            request_id=job.request_id,
            dispatch_payload=job.dispatch_payload,
            payload_version=job.payload_version,
            deduplication_key=job.deduplication_key,
            progress=job.progress,
            # Kept out of the payload contract: relays and task handlers of any version ignore it.
            trace_context=current_trace_carrier() or None,
        )
        self._db.add(orm)
        try:
            await self._db.flush()
        except IntegrityError as exc:
            if "ux_background_jobs_active_dedup" not in str(
                exc
            ) and "background_jobs.deduplication_key" not in str(exc):
                raise

            message = "An equivalent ingestion is already queued or running"
            raise BusinessRuleViolationError(message) from exc
        await self._db.refresh(orm)
        job.id = orm.id
        job.creation_date = orm.creation_date
        return job

    async def check_admission(self, principal: object) -> None:
        # Keep each principal's count + insert exact across API replicas until commit.
        # The global overload guard is approximate: unrelated uploads must not queue here.
        if isinstance(principal, str) and self._db.get_bind().dialect.name == "postgresql":
            key = f"background_job_admission:{principal}"
            await self._db.execute(select(func.pg_advisory_xact_lock(func.hashtextextended(key, 0))))

        settings = process_section(JobsSection)
        active = BackgroundJobModel.status.in_(["pending", "running"])
        count = await self._db.scalar(select(func.count()).select_from(BackgroundJobModel).where(active))

        if (count or 0) >= settings.job_backlog_limit:
            scope = "global"
            raise JobAdmissionError(scope)

        if isinstance(principal, str):
            count = await self._db.scalar(
                select(func.count())
                .select_from(BackgroundJobModel)
                .where(active, BackgroundJobModel.dispatch_payload["cost_principal"].as_string() == principal)
            )

            if (count or 0) >= settings.job_principal_backlog_limit:
                scope = "principal"
                raise JobAdmissionError(scope)

    async def pending_dispatch(self, limit: int = 100) -> list[BackgroundJob]:
        """Lock pending durable submissions until the caller commits or rolls back."""
        now = datetime.now(UTC)
        ranked = (
            select(
                BackgroundJobModel.id,
                func.row_number()
                .over(
                    partition_by=BackgroundJobModel.dispatch_payload["cost_principal"].as_string(),
                    order_by=(BackgroundJobModel.heartbeat_at.asc().nulls_first(), BackgroundJobModel.id),
                )
                .label("position"),
            )
            .where(
                BackgroundJobModel.status == BackgroundJobStatus.PENDING.value,
                BackgroundJobModel.dispatch_payload.is_not(None),
                or_(BackgroundJobModel.next_attempt_at.is_(None), BackgroundJobModel.next_attempt_at <= now),
                or_(
                    BackgroundJobModel.heartbeat_at.is_(None),
                    BackgroundJobModel.heartbeat_at <= now - REDISPATCH_AFTER,
                ),
            )
            .subquery()
        )
        result = await self._db.execute(
            select(BackgroundJobModel)
            .join(ranked, ranked.c.id == BackgroundJobModel.id)
            .where(
                BackgroundJobModel.status == BackgroundJobStatus.PENDING.value,
                BackgroundJobModel.dispatch_payload.is_not(None),
                or_(BackgroundJobModel.next_attempt_at.is_(None), BackgroundJobModel.next_attempt_at <= now),
                or_(
                    BackgroundJobModel.heartbeat_at.is_(None),
                    BackgroundJobModel.heartbeat_at <= now - REDISPATCH_AFTER,
                ),
            )
            .order_by(
                ranked.c.position, BackgroundJobModel.heartbeat_at.asc().nulls_first(), BackgroundJobModel.id
            )
            .limit(limit)
            .with_for_update(of=BackgroundJobModel, skip_locked=True)
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

    async def delete_old(self, days: int, limit: int = 1000) -> int:
        cutoff = datetime.now(tz=UTC) - timedelta(days=days)
        ids = (
            select(BackgroundJobModel.id)
            .where(
                BackgroundJobModel.creation_date < cutoff,
                BackgroundJobModel.status.in_(["done", "failed", "cancelled"]),
            )
            .order_by(BackgroundJobModel.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = await self._db.execute(
            delete(BackgroundJobModel)
            .where(BackgroundJobModel.id.in_(ids))
            .execution_options(synchronize_session=False)
        )
        await self._db.flush()
        return result.rowcount or 0

    async def list_recent(
        self, limit: int = 50, offset: int = 0, status: str | None = None
    ) -> list[BackgroundJob]:
        result = await self._db.execute(
            select(BackgroundJobModel)
            .where(BackgroundJobModel.status == status if status is not None else true())
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

    async def find_latest_for(self, job_type: str, related_id: int) -> BackgroundJob | None:
        result = await self._db.execute(
            select(BackgroundJobModel)
            .where(BackgroundJobModel.job_type == job_type, BackgroundJobModel.related_id == related_id)
            .order_by(BackgroundJobModel.id.desc())
            .limit(1)
        )
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm is not None else None

    async def count_by_status(self) -> dict[str, int]:
        result = await self._db.execute(
            select(BackgroundJobModel.status, func.count()).group_by(BackgroundJobModel.status)
        )
        return {row[0]: row[1] for row in result.all()}

    async def lock_execution(self, job_id: int, attempt_id: str) -> None:
        with measure_fencing():
            result = await self._db.execute(
                select(BackgroundJobModel.id)
                .where(
                    BackgroundJobModel.id == job_id,
                    BackgroundJobModel.attempt_id == attempt_id,
                    BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value,
                    BackgroundJobModel.lease_until > datetime.now(UTC),
                )
                # SHARE blocks attempt UPDATE but permits parallel UoWs of this owner.
                .with_for_update(read=True)
            )

        if result.scalar_one_or_none() is None:
            message = "Job execution lease is no longer current"
            raise ExecutionLeaseLostError(message)

    async def claim(
        self, job_id: int, attempt_id: str, *, lease_seconds: int, limit: ConcurrencyLimit | None = None
    ) -> int | None:
        """Start a new attempt of a due PENDING job; expired leases are released by recovery.

        With ``limit``, claims of the group are serialized for this transaction and
        the job waits (``JobConcurrencyLimitReachedError``) while the group is full.
        """
        now = datetime.now(UTC)
        claimable = (
            BackgroundJobModel.id == job_id,
            BackgroundJobModel.status == BackgroundJobStatus.PENDING.value,
            or_(BackgroundJobModel.next_attempt_at.is_(None), BackgroundJobModel.next_attempt_at <= now),
        )

        if limit is not None:
            await self.serialize_claims(limit)

            if await self._db.scalar(select(BackgroundJobModel.id).where(*claimable)) is None:
                return None

            running = await self._db.scalar(
                select(func.count())
                .select_from(BackgroundJobModel)
                .where(
                    BackgroundJobModel.job_type.in_(limit.job_types),
                    BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value,
                    BackgroundJobModel.lease_until > now,
                )
            )

            if limit.max_running is not None and (running or 0) >= limit.max_running:
                message = f"{running} of {limit.max_running} slots busy"
                raise JobConcurrencyLimitReachedError(message)

            if limit.max_per_principal is not None:
                principal = await self._db.scalar(
                    select(BackgroundJobModel.dispatch_payload["cost_principal"].as_string()).where(
                        BackgroundJobModel.id == job_id
                    )
                )
                running = await self._db.scalar(
                    select(func.count())
                    .select_from(BackgroundJobModel)
                    .where(
                        BackgroundJobModel.job_type.in_(limit.job_types),
                        BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value,
                        BackgroundJobModel.lease_until > now,
                        BackgroundJobModel.dispatch_payload["cost_principal"].as_string() == principal,
                    )
                )

                if (running or 0) >= limit.max_per_principal:
                    message = "Principal concurrency limit reached"
                    raise JobConcurrencyLimitReachedError(message)

        result = await self._db.execute(
            update(BackgroundJobModel)
            .where(*claimable)
            .values(
                status=BackgroundJobStatus.RUNNING.value,
                attempt_id=attempt_id,
                attempt=BackgroundJobModel.attempt + 1,
                started_at=now,
                heartbeat_at=now,
                lease_until=now + timedelta(seconds=lease_seconds),
                next_attempt_at=None,
                error_message=None,
            )
            .returning(BackgroundJobModel.attempt)
        )
        return result.scalar_one_or_none()

    async def cancel(self, job_id: int) -> bool:
        result = await self._db.execute(
            update(BackgroundJobModel)
            .where(
                BackgroundJobModel.id == job_id,
                BackgroundJobModel.status.in_(["pending", "running"]),
            )
            .values(
                status=BackgroundJobStatus.CANCELLED.value,
                finished_at=datetime.now(UTC),
                attempt_id=None,
                lease_until=None,
                next_attempt_at=None,
            )
        )
        return bool(result.rowcount)

    async def save_progress(self, job_id: int, progress: dict[str, str]) -> None:
        await self._db.execute(
            update(BackgroundJobModel)
            .where(
                BackgroundJobModel.id == job_id,
                BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value,
            )
            .values(progress=progress)
        )

    async def serialize_claims(self, limit: ConcurrencyLimit) -> None:
        # Count-then-claim is racy under READ COMMITTED; a transaction-scoped
        # advisory lock per group makes it atomic across worker replicas.
        if self._db.get_bind().dialect.name != "postgresql":
            return

        key = "background_job_claim:" + ",".join(sorted(limit.job_types))
        await self._db.execute(select(func.pg_advisory_xact_lock(func.hashtext(key))))

    async def renew_lease(self, job_id: int, attempt_id: str, *, lease_seconds: int) -> bool:
        now = datetime.now(UTC)
        result = await self._db.execute(
            update(BackgroundJobModel)
            .where(
                BackgroundJobModel.id == job_id,
                BackgroundJobModel.attempt_id == attempt_id,
                BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value,
                BackgroundJobModel.lease_until > now,
            )
            .values(heartbeat_at=now, lease_until=now + timedelta(seconds=lease_seconds))
        )
        return bool(result.rowcount)

    async def finish_attempt(
        self, job_id: int, attempt_id: str, *, error: str | None = None, retry_seconds: float | None = None
    ) -> bool:
        now = datetime.now(UTC)
        status = BackgroundJobStatus.DONE if error is None else BackgroundJobStatus.FAILED

        if retry_seconds is not None:
            status = BackgroundJobStatus.PENDING

        result = await self._db.execute(
            update(BackgroundJobModel)
            .where(
                BackgroundJobModel.id == job_id,
                BackgroundJobModel.attempt_id == attempt_id,
                BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value,
                BackgroundJobModel.lease_until > now,
            )
            .values(
                status=status.value,
                error_message=error,
                lease_until=None,
                attempt_id=None,
                finished_at=None if retry_seconds is not None else now,
                next_attempt_at=now + timedelta(seconds=retry_seconds) if retry_seconds is not None else None,
                # A retry is due for relay publication as soon as its delay passes.
                heartbeat_at=None if retry_seconds is not None else BackgroundJobModel.heartbeat_at,
            )
        )
        return bool(result.rowcount)

    async def record_dispatch(self, job_id: int, *, published: bool) -> None:
        """Rotate relay order; count only publications the broker accepted as new."""
        values: dict[str, object] = {"heartbeat_at": datetime.now(UTC)}

        if published:
            values["dispatch_attempts"] = BackgroundJobModel.dispatch_attempts + 1

        await self._db.execute(
            update(BackgroundJobModel).where(BackgroundJobModel.id == job_id).values(**values)
        )

    async def indexing_queue_snapshot(self) -> IndexingQueueSnapshot:
        now = datetime.now(UTC)
        pending = BackgroundJobModel.status == BackgroundJobStatus.PENDING.value
        running = BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value
        ready_since = func.coalesce(BackgroundJobModel.next_attempt_at, BackgroundJobModel.creation_date)
        pending_count, running_count, oldest, expired = (
            await self._db.execute(
                select(
                    func.count().filter(pending),
                    func.count().filter(running),
                    func.min(ready_since).filter(pending, ready_since <= now),
                    func.count().filter(
                        running,
                        expired_execution(now, process_section(JobsSection).stuck_job_timeout_minutes),
                    ),
                ).where(
                    BackgroundJobModel.job_type.in_(
                        (JobType.DOCUMENT_PROCESSING, JobType.INGEST, JobType.INGEST_FILE)
                    ),
                    or_(pending, running),
                )
            )
        ).one()
        return IndexingQueueSnapshot(
            pending=pending_count,
            running=running_count,
            oldest_ready_seconds=max(0, int((now - oldest).total_seconds())) if oldest is not None else None,
            expired_leases=expired,
        )

    async def lock_expired(self, timeout_minutes: int, limit: int = 100) -> list[BackgroundJob]:
        """Lock RUNNING jobs whose owner stopped renewing; concurrent recoveries skip them."""
        now = datetime.now(UTC)
        result = await self._db.execute(
            select(BackgroundJobModel)
            .where(
                BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value,
                expired_execution(now, timeout_minutes),
            )
            .order_by(BackgroundJobModel.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def release_expired(self, job_id: int, *, error: str, retry_seconds: float | None) -> None:
        """Fence out the lost attempt, then requeue it (``retry_seconds``) or fail it."""
        now = datetime.now(UTC)
        retry = retry_seconds is not None
        await self._db.execute(
            update(BackgroundJobModel)
            .where(
                BackgroundJobModel.id == job_id,
                BackgroundJobModel.status == BackgroundJobStatus.RUNNING.value,
            )
            .values(
                status=(BackgroundJobStatus.PENDING if retry else BackgroundJobStatus.FAILED).value,
                attempt_id=None,
                lease_until=None,
                error_message=error[:500],
                finished_at=None if retry else now,
                next_attempt_at=now + timedelta(seconds=retry_seconds) if retry_seconds is not None else None,
                # Due for immediate relay publication once the retry delay passes.
                heartbeat_at=None if retry else BackgroundJobModel.heartbeat_at,
            )
        )

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
