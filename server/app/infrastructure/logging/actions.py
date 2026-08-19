"""User action logger -- records meaningful operations to a dedicated audit log.

Emits structured log entries (via the ``actions`` logger) for operations
such as document uploads, user creation, and config changes.  Entries
include the action name, user ID, and relevant metadata for compliance
and debugging.

When ``LOG_FORMAT=json``, structured fields are passed via ``extra`` so
that the JSON formatter emits them as separate top-level keys (``action``,
``action_user_id``, ``action_details``) instead of flattening them into
the message string.
"""

from __future__ import annotations

import logging
from typing import Any

_action_logger = logging.getLogger("actions")


def log_action(action: str, user_id: int | None = None, details: dict[str, Any] | None = None) -> None:
    """Log a user action with structured fields.

    Human-readable message stays as before (for text-mode logs and the
    in-memory buffer).  Structured fields are passed via ``extra`` so the
    JSON formatter can emit them as separate keys when LOG_FORMAT=json.

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

    extra: dict[str, Any] = {"action": action, "action_user_id": user_id, "action_details": details or {}}
    _action_logger.info(" | ".join(parts), extra=extra)
