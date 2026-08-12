"""ChatLogRepository — persistence protocol for Q&A quality logs."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from domain.entities.chat_log import ChatLog


class ChatLogRepository(Protocol):
    async def save(self, log: ChatLog) -> None: ...

    async def list_logs(
        self,
        *,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatLog]: ...

    async def count_logs(
        self,
        *,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
    ) -> int: ...
