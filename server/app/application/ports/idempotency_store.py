"""Protocol for idempotency key store — prevents duplicate write executions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IdempotencyStorePort(Protocol):
    """Store for idempotency keys (duplicate-request prevention)."""

    async def get(self, key: str, user_id: int) -> dict | None: ...
    async def store(self, key: str, user_id: int, result: dict[str, Any]) -> None: ...
