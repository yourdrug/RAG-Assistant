"""ApiKeyProvider port — abstract interface for API key operations.

Lives in the application layer so that auth_service depends on a port,
not on the concrete Redis-backed implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ApiKeyProviderPort(Protocol):
    def generate_key(self) -> str: ...
    def hash_key(self, raw_key: str) -> str: ...
    def key_prefix_for_display(self, raw_key: str) -> str: ...
    async def invalidate_by_id(self, api_key_id: int) -> None: ...
