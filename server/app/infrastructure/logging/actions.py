"""User action logger -- records meaningful operations to a dedicated audit log.

Emits structured log entries (via the ``actions`` logger) for operations
such as document uploads, user creation, and config changes.  Entries
include the action name, user ID, and relevant metadata for compliance
and debugging.
"""

from __future__ import annotations

import logging
from typing import Any

_action_logger = logging.getLogger("actions")


def log_action(action: str, user_id: int | None = None, details: dict[str, Any] | None = None) -> None:
    """Log a user action.

    Args:
        action: Short action name, e.g. "login", "chat", "document.upload"
        user_id: ID of the user performing the action
        details: Optional extra context (filename, question preview, etc.)
    """
    parts = [action]
    if user_id is not None:
        parts.append(f"user={user_id}")
    if details:
        for k, v in details.items():
            parts.append(f"{k}={v}")
    _action_logger.info(" | ".join(parts))
