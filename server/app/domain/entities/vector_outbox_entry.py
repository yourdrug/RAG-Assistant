"""VectorOutboxEntry entity — a pending Qdrant operation stored in Postgres.

Part of the Transactional Outbox pattern ensuring Postgres ↔ Qdrant consistency.
Each entry represents a single operation (upsert or delete) that must be applied
to the vector store after the owning Postgres transaction commits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class OutboxOperation(StrEnum):
    UPSERT_CHUNKS = "upsert_chunks"
    DELETE_BY_DOCUMENT = "delete_by_document"
    DELETE_CHUNKS = "delete_chunks"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class VectorOutboxEntry:
    id: int | None = None
    operation: OutboxOperation = OutboxOperation.UPSERT_CHUNKS
    aggregate_type: str = "document"
    aggregate_id: int = 0
    payload: dict = field(default_factory=dict)
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    max_attempts: int = 8
    last_error: str | None = None
    creation_date: datetime | None = None
