"""SQLAlchemy ORM implementation of VectorOutboxRepository.

Implements the Transactional Outbox pattern for Postgres ↔ Qdrant consistency.
The enqueue() method sends pg_notify atomically within the same transaction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from domain.entities.vector_outbox_entry import OutboxOperation, OutboxStatus, VectorOutboxEntry
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import VectorStoreOutboxModel


class SQLAlchemyVectorOutboxRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def enqueue(self, entry: VectorOutboxEntry) -> VectorOutboxEntry:
        orm = VectorStoreOutboxModel(
            operation=entry.operation.value,
            aggregate_type=entry.aggregate_type,
            aggregate_id=entry.aggregate_id,
            payload=entry.payload,
            status=OutboxStatus.PENDING.value,
            next_attempt_at=datetime.now(tz=UTC),
        )
        self._db.add(orm)
        await self._db.flush()

        # Atomic NOTIFY — same transaction as the INSERT
        await self._db.execute(
            text("SELECT pg_notify(:channel, :payload)"),
            {
                "channel": "vector_outbox_ready",
                "payload": json.dumps({"id": orm.id, "op": entry.operation.value}),
            },
        )

        entry.id = orm.id
        return entry

    async def claim_batch(self, worker_id: str, limit: int = 20) -> list[VectorOutboxEntry]:
        """Atomically claim a batch using SELECT ... FOR UPDATE SKIP LOCKED."""
        rows = await self._db.execute(
            text(
                """
                UPDATE vector_store_outbox
                SET status = 'in_progress',
                    locked_by = :worker_id,
                    locked_at = NOW()
                WHERE id IN (
                    SELECT id FROM vector_store_outbox
                    WHERE status IN ('pending', 'failed')
                      AND next_attempt_at <= NOW()
                    ORDER BY id
                    LIMIT :limit
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, operation, aggregate_type, aggregate_id, payload,
                          status, attempts, max_attempts, last_error, creation_date
                """
            ),
            {"worker_id": worker_id, "limit": limit},
        )
        return [self._row_to_entity(r) for r in rows.mappings().all()]

    async def recover_stuck(self, stuck_timeout_minutes: int = 5) -> int:
        """Reset in_progress entries stuck for longer than stuck_timeout_minutes.

        Handles crash recovery: dispatcher claimed entries but died before mark_done/mark_failed.
        Resets to 'pending' so they'll be retried on the next dispatch cycle.
        Returns the number of entries recovered.
        """
        from datetime import timedelta

        cutoff = datetime.now(tz=UTC) - timedelta(minutes=stuck_timeout_minutes)
        result = await self._db.execute(
            select(VectorStoreOutboxModel).where(
                VectorStoreOutboxModel.status == OutboxStatus.IN_PROGRESS.value,
                VectorStoreOutboxModel.locked_at < cutoff,
            )
        )
        orms = result.scalars().all()
        for orm in orms:
            orm.status = OutboxStatus.PENDING.value
            orm.locked_by = None
            orm.locked_at = None
            orm.last_error = "Recovered from stuck in_progress (dispatcher crash)"
        if orms:
            await self._db.flush()
        return len(orms)

    async def mark_done(self, entry_id: int) -> None:
        result = await self._db.execute(
            select(VectorStoreOutboxModel).where(VectorStoreOutboxModel.id == entry_id)
        )
        orm = result.scalar_one_or_none()
        if orm:
            orm.status = OutboxStatus.DONE.value
            orm.completed_at = datetime.now(tz=UTC)
            await self._db.flush()

    async def mark_failed(self, entry_id: int, error: str, backoff_seconds: int) -> None:
        result = await self._db.execute(
            select(VectorStoreOutboxModel).where(VectorStoreOutboxModel.id == entry_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return
        orm.attempts += 1
        orm.last_error = error[:2000]
        if orm.attempts >= orm.max_attempts:
            orm.status = OutboxStatus.DEAD_LETTER.value
        else:
            orm.status = OutboxStatus.FAILED.value
            orm.next_attempt_at = datetime.now(tz=UTC) + timedelta(seconds=backoff_seconds)
        await self._db.flush()

    async def mark_dead_letter(self, entry_id: int, error: str) -> None:
        result = await self._db.execute(
            select(VectorStoreOutboxModel).where(VectorStoreOutboxModel.id == entry_id)
        )
        orm = result.scalar_one_or_none()
        if orm:
            orm.status = OutboxStatus.DEAD_LETTER.value
            orm.last_error = error[:2000]
            await self._db.flush()

    async def count_pending(self) -> int:
        result = await self._db.execute(
            select(func.count())
            .select_from(VectorStoreOutboxModel)
            .where(VectorStoreOutboxModel.status.in_(["pending", "failed", "in_progress"]))
        )
        return result.scalar_one() or 0

    async def count_by_document(self, document_id: int) -> dict[str, int]:
        pending_statuses = ["pending", "failed", "in_progress"]
        failed_statuses = ["failed", "dead_letter"]

        pending_result = await self._db.execute(
            select(func.count())
            .select_from(VectorStoreOutboxModel)
            .where(
                VectorStoreOutboxModel.aggregate_id == document_id,
                VectorStoreOutboxModel.status.in_(pending_statuses),
            )
        )
        pending = pending_result.scalar_one() or 0

        failed_result = await self._db.execute(
            select(func.count())
            .select_from(VectorStoreOutboxModel)
            .where(
                VectorStoreOutboxModel.aggregate_id == document_id,
                VectorStoreOutboxModel.status.in_(failed_statuses),
            )
        )
        failed = failed_result.scalar_one() or 0

        return {"pending": pending, "failed": failed}

    async def get_failed_details(self, document_id: int) -> list[dict[str, object]]:
        """Return failed/dead_letter entries with retry info for a document."""
        failed_statuses = ["failed", "dead_letter"]
        result = await self._db.execute(
            select(
                VectorStoreOutboxModel.operation,
                VectorStoreOutboxModel.attempts,
                VectorStoreOutboxModel.max_attempts,
                VectorStoreOutboxModel.last_error,
            ).where(
                VectorStoreOutboxModel.aggregate_id == document_id,
                VectorStoreOutboxModel.status.in_(failed_statuses),
            )
        )
        return [
            {
                "operation": row.operation,
                "attempts": row.attempts,
                "max_attempts": row.max_attempts,
                "last_error": row.last_error,
            }
            for row in result.mappings().all()
        ]

    async def list_dead_letters(self, limit: int = 50) -> list[VectorOutboxEntry]:
        result = await self._db.execute(
            select(VectorStoreOutboxModel)
            .where(VectorStoreOutboxModel.status == OutboxStatus.DEAD_LETTER.value)
            .order_by(VectorStoreOutboxModel.creation_date.desc())
            .limit(limit)
        )
        return [self._orm_to_entity(o) for o in result.scalars().all()]

    @staticmethod
    def _row_to_entity(row) -> VectorOutboxEntry:
        return VectorOutboxEntry(
            id=row["id"],
            operation=OutboxOperation(row["operation"]),
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            payload=row["payload"],
            status=OutboxStatus(row["status"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            last_error=row["last_error"],
            creation_date=row["creation_date"],
        )

    @staticmethod
    def _orm_to_entity(orm: VectorStoreOutboxModel) -> VectorOutboxEntry:
        return VectorOutboxEntry(
            id=orm.id,
            operation=OutboxOperation(orm.operation),
            aggregate_type=orm.aggregate_type,
            aggregate_id=orm.aggregate_id,
            payload=orm.payload,
            status=OutboxStatus(orm.status),
            attempts=orm.attempts,
            max_attempts=orm.max_attempts,
            last_error=orm.last_error,
            creation_date=orm.creation_date,
        )
