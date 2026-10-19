"""LogBufferPort — reads recent log entries for admin dashboard."""

from __future__ import annotations

from typing import Any, Protocol


class LogBufferPort(Protocol):
    """Provides access to recent application log entries."""

    def get_logs(
        self,
        *,
        limit: int = 100,
        level: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]: ...
