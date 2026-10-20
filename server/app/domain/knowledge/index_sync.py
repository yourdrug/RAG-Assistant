"""Knowledge domain: index sync."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from domain.knowledge.chunk import Chunk
from domain.shared.enums import LifecycleStatusMixin


class OutboxOperation(StrEnum):
    UPSERT_CHUNKS = "upsert_chunks"
    DELETE_BY_DOCUMENT = "delete_by_document"
    DELETE_CHUNKS = "delete_chunks"
    RENAME_DOCUMENT = "rename_document"
    UPDATE_METADATA = "update_metadata"
    SET_DOCUMENT_ID = "set_document_id"


class OutboxStatus(
    LifecycleStatusMixin,
    StrEnum,
    terminal=frozenset({"done", "dead_letter"}),
    active=frozenset({"pending", "in_progress"}),
):
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

    def mark_running(self) -> None:
        """Transition to IN_PROGRESS status."""
        self.status = OutboxStatus.IN_PROGRESS

    def mark_done(self) -> None:
        """Transition to DONE status."""
        self.status = OutboxStatus.DONE

    def mark_failed(self, error: str) -> None:
        """Record a failure attempt.  Check ``is_dead_letter`` after calling."""
        self.attempts += 1
        self.last_error = error
        if self.attempts >= self.max_attempts:
            self.status = OutboxStatus.DEAD_LETTER
        else:
            self.status = OutboxStatus.FAILED

    @property
    def is_dead_letter(self) -> bool:
        """True if max attempts exceeded -- entry needs manual intervention."""
        return self.status == OutboxStatus.DEAD_LETTER


@runtime_checkable
class OutboxProducer(Protocol):
    """Write-side: enqueue vector-store operations."""

    async def enqueue(self, entry: VectorOutboxEntry) -> VectorOutboxEntry: ...


@runtime_checkable
class OutboxDispatcher(Protocol):
    """Dispatch-side: claim, complete, fail, and recover outbox entries."""

    async def claim_batch(self, worker_id: str, limit: int = 20) -> list[VectorOutboxEntry]: ...

    async def mark_done(self, entry_id: int) -> None: ...

    async def mark_failed(self, entry_id: int, error: str, backoff_seconds: int) -> None: ...

    async def mark_dead_letter(self, entry_id: int, error: str) -> None: ...

    async def count_pending(self) -> int: ...

    async def recover_stuck(self, stuck_timeout_minutes: int = 5) -> int: ...


@runtime_checkable
class OutboxInspector(Protocol):
    """Read-side: monitoring and dead-letter inspection."""

    async def count_by_document(self, document_id: int) -> dict[str, int]: ...

    async def count_by_documents(self, document_ids: list[int]) -> dict[int, dict[str, int]]: ...

    async def get_failed_details(self, document_id: int) -> list[dict[str, object]]: ...

    async def get_failed_details_by_documents(
        self, document_ids: list[int]
    ) -> dict[int, list[dict[str, object]]]: ...

    async def list_dead_letters(self, limit: int = 50) -> list[VectorOutboxEntry]: ...


@runtime_checkable
class VectorOutboxRepository(OutboxProducer, OutboxDispatcher, OutboxInspector, Protocol):
    """Combined protocol -- the UnitOfWork attribute type.

    Services should prefer the narrow protocols (``OutboxProducer``,
    ``OutboxDispatcher``, ``OutboxInspector``) for their constructor type hints.
    """


@runtime_checkable
class VectorIndexWriter(Protocol):
    async def ensure_collection(self, vector_size: int, reset: bool = False) -> None: ...
    async def upload_documents(
        self,
        chunks: list[Chunk],
        *,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
    ) -> None: ...
    async def delete_by_document_id(self, document_id: int) -> None: ...
    async def upsert_point(self, point_id: int, vector: list[float], payload: dict) -> None: ...
    async def delete_by_ids(self, ids: list[int]) -> None: ...
    async def set_document_id_by_source(self, source: str, document_id: int) -> None: ...
    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> None: ...
    async def update_metadata_by_act_version(self, act_version_id: int, metadata_updates: dict) -> None: ...
