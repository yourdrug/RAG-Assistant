"""Vector outbox repository interface —Transactional Outbox for Postgres ↔ Qdrant consistency.

The outbox stores pending vector-store operations atomically with Postgres
mutations.  A background dispatcher reads and applies them to Qdrant.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.vector_outbox_entry import VectorOutboxEntry


@runtime_checkable
class VectorOutboxRepository(Protocol):
    async def enqueue(self, entry: VectorOutboxEntry) -> VectorOutboxEntry:
        """Insert a pending outbox entry. Sends pg_notify atomically within the same transaction."""
        ...

    async def claim_batch(self, worker_id: str, limit: int = 20) -> list[VectorOutboxEntry]:
        """Atomically claim a batch of pending/failed entries ready for processing.

        Uses SELECT ... FOR UPDATE SKIP LOCKED so multiple dispatcher instances
        can run in parallel without duplicating work.
        """
        ...

    async def mark_done(self, entry_id: int) -> None:
        """Mark an entry as successfully applied to the vector store."""
        ...

    async def mark_failed(self, entry_id: int, error: str, backoff_seconds: int) -> None:
        """Mark an entry as failed and schedule retry with backoff.

        Automatically promotes to dead_letter when max_attempts is reached.
        """
        ...

    async def mark_dead_letter(self, entry_id: int, error: str) -> None:
        """Move an entry to dead letter queue for manual inspection."""
        ...

    async def count_pending(self) -> int:
        """Count entries still awaiting processing (pending + failed + in_progress)."""
        ...

    async def recover_stuck(self, stuck_timeout_minutes: int = 5) -> int:
        """Reset in_progress entries stuck for longer than timeout (dispatcher crash recovery)."""
        ...

    async def count_by_document(self, document_id: int) -> dict[str, int]:
        """Count outbox entries for a specific document by status.

        Returns {"pending": N, "failed": M} where:
        - pending = entries in pending/failed/in_progress (not yet applied)
        - failed = entries in failed/dead_letter (need attention)
        """
        ...

    async def get_failed_details(self, document_id: int) -> list[dict[str, object]]:
        """Return failed/dead_letter entries with retry info for a document.

        Each dict contains: operation, attempts, max_attempts, last_error.
        """
        ...

    async def list_dead_letters(self, limit: int = 50) -> list[VectorOutboxEntry]:
        """List dead-lettered entries for manual inspection."""
        ...
