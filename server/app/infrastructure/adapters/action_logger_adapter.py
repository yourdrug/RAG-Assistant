"""Infrastructure adapter for ActionLoggerPort — wraps logging.actions.log_action."""

from __future__ import annotations

from typing import Any

from infrastructure.logging.actions import log_action as _log_action


class ActionLoggerAdapter:
    """Thin wrapper making log_action available as an injectable port."""

    def __call__(
        self,
        action: str,
        *,
        user_id: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        _log_action(action, user_id=user_id, details=details)
