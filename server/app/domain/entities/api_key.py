"""ApiKey Entity — статический ключ доступа для внешних (client) пользователей."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ApiKey:
    id: int | None = None
    user_id: int = 0
    key_hash: str = ""
    key_prefix: str = ""
    name: str | None = None
    created_at: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None
