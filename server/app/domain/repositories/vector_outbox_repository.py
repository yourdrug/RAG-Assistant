"""Vector outbox repository interface -- Transactional Outbox for Postgres <-> Qdrant consistency.

Three narrow protocols (Producer, Dispatcher, Inspector) cover distinct
consumer roles.  The combined ``VectorOutboxRepository`` inherits all three
so the UnitOfWork can expose a single attribute.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.vector_outbox_entry import VectorOutboxEntry


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

    async def get_failed_details(self, document_id: int) -> list[dict[str, object]]: ...

    async def list_dead_letters(self, limit: int = 50) -> list[VectorOutboxEntry]: ...


@runtime_checkable
class VectorOutboxRepository(OutboxProducer, OutboxDispatcher, OutboxInspector, Protocol):
    """Combined protocol -- the UnitOfWork attribute type.

    Services should prefer the narrow protocols (``OutboxProducer``,
    ``OutboxDispatcher``, ``OutboxInspector``) for their constructor type hints.
    """

    ...
