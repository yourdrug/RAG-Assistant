"""In-memory log buffer for UI consumption."""

from __future__ import annotations

import logging
import threading
from collections import deque


class LogBufferHandler(logging.Handler):
    """Logging handler that stores log records in a ring buffer."""

    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self._buffer: deque[dict] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": self.format(record),
                "level": record.levelname,
                "logger": record.name,
                "request_id": getattr(record, "request_id", "-"),
                "message": record.getMessage(),
                "filename": getattr(record, "filename", None),
                "lineno": getattr(record, "lineno", None),
            }
            with self._lock:
                self._buffer.append(entry)
        except Exception:
            pass

    def get_logs(
        self,
        limit: int = 100,
        level: str | None = None,
        search: str | None = None,
    ) -> list[dict]:
        with self._lock:
            logs = list(self._buffer)

        if level:
            level = level.upper()
            logs = [entry for entry in logs if entry["level"] == level]

        if search:
            search_lower = search.lower()
            logs = [entry for entry in logs if search_lower in entry["message"].lower()]

        return list(reversed(logs[-limit:]))


log_buffer = LogBufferHandler(capacity=2000)
log_buffer.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)


def attach_log_buffer() -> None:
    """Attach the log buffer handler to the 'actions' logger only."""
    actions_logger = logging.getLogger("actions")
    actions_logger.setLevel(logging.INFO)
    actions_logger.addHandler(log_buffer)
    actions_logger.propagate = False
