"""ApiKey Entity -- static access key for external (client) users."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class ApiKey:
    id: int | None = None
    user_id: int = 0
    key_hash: str = ""
    key_prefix: str = ""
    name: str | None = None
    creation_date: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


@dataclass(frozen=True)
class ApiKeyClientInfo:
    """Combined ApiKey + User data returned by get_active_client_by_hash."""

    api_key_id: int
    id: int
    email: str
    role: str
    kind: str
    is_active: bool
