"""ApiKey Repository interface."""

from __future__ import annotations

from typing import Protocol

from domain.entities.api_key import ApiKey


class ApiKeyRepository(Protocol):
    async def create(self, user_id: int, key_hash: str, key_prefix: str, name: str | None = None) -> ApiKey: ...

    async def list_for_user(self, user_id: int) -> list[ApiKey]: ...

    async def revoke(self, api_key_id: int, user_id: int | None = None) -> bool: ...

    async def get_active_client_by_hash(self, key_hash: str) -> dict | None:
        """Вернуть данные пользователя для активного ключа, ТОЛЬКО если владелец kind='client'."""
        ...

    async def touch_last_used(self, api_key_id: int) -> None: ...
