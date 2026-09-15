"""Infrastructure adapter for LogBufferPort — wraps logging.log_buffer."""

from __future__ import annotations

from typing import Any

from infrastructure.logging.log_buffer import log_buffer as _log_buffer


class LogBufferAdapter:
    """Thin wrapper making log buffer available as an injectable port."""

    def get_logs(
        self,
        *,
        limit: int = 100,
        level: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        return _log_buffer.get_logs(limit=limit, level=level, search=search)
