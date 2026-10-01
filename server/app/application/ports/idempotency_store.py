"""Protocol for idempotency key store — prevents duplicate write executions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class IdempotencyClaimStatus(StrEnum):
    """States returned by an idempotency reservation claim."""

    ACQUIRED = "acquired"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CONFLICT = "conflict"


@runtime_checkable
class IdempotencyStorePort(Protocol):
    """Store for idempotency keys (duplicate-request prevention)."""

    async def claim(
        self, key: str, principal_id: str, operation: str, fingerprint: str
    ) -> dict[str, Any]: ...

    async def complete(self, reservation: dict[str, Any], result: dict[str, Any]) -> None: ...
