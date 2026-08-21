"""In-memory cache for dry-run preview files with TTL-based expiration."""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path

logger = logging.getLogger("default")

DEFAULT_TTL_SECONDS = 1800  # 30 minutes


class PreviewCache:
    """Stores temporary PDF files for dry-run preview sessions.

    Each file is assigned a unique ``preview_id`` (UUID4). Files are
    automatically evicted after *ttl_seconds*.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._cache: dict[str, tuple[Path, float]] = {}
        self._ttl = ttl_seconds

    def store(self, file_data: bytes, suffix: str = ".pdf") -> str:
        """Persist *file_data* to a temp file and return its ``preview_id``."""
        preview_id = uuid.uuid4().hex
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(file_data)
        tmp.close()
        self._cache[preview_id] = (Path(tmp.name), time.monotonic())
        self._cleanup_expired()
        return preview_id

    def get_path(self, preview_id: str) -> Path | None:
        """Return the cached file path if still valid, else ``None``."""
        entry = self._cache.get(preview_id)
        if entry is None:
            return None
        path, ts = entry
        if time.monotonic() - ts > self._ttl:
            self._evict(preview_id)
            return None
        if not path.exists():
            self._cache.pop(preview_id, None)
            return None
        return path

    def delete(self, preview_id: str) -> None:
        """Explicitly remove a cached entry."""
        self._evict(preview_id)

    # ------------------------------------------------------------------

    def _evict(self, preview_id: str) -> None:
        entry = self._cache.pop(preview_id, None)
        if entry is not None:
            path, _ = entry
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to delete cached preview file: %s", path)

    def _cleanup_expired(self) -> None:
        now = time.monotonic()
        expired = [pid for pid, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for pid in expired:
            self._evict(pid)
