"""ActionLoggerPort — audit logging port for presentation layer."""

from __future__ import annotations

from typing import Any, Protocol


class ActionLoggerPort(Protocol):
    """Logs user actions for audit trail."""

    def __call__(
        self,
        action: str,
        *,
        user_id: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None: ...
