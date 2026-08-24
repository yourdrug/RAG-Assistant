"""S3-backed cache for dry-run preview files with TTL-based expiration.

Stores preview PDFs in the configured ``FileStorage`` backend (S3 or local)
under the ``previews/`` prefix.  An in-memory dict tracks keys and
timestamps for fast TTL checks; stale objects are evicted on each
``store()`` call.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.ports.file_storage import FileStorage

logger = logging.getLogger("default")

DEFAULT_TTL_SECONDS = 1800  # 30 minutes
PREVIEW_PREFIX = "previews/"


class PreviewCache:
    """Stores PDF files for dry-run preview sessions in FileStorage.

    Each file is assigned a unique ``preview_id`` (UUID4 hex).  Files are
    automatically evicted after *ttl_seconds*.
    """

    def __init__(self, storage: FileStorage, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._storage = storage
        self._cache: dict[str, tuple[str, float]] = {}  # preview_id -> (s3_key, timestamp)
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store(self, file_data: bytes, suffix: str = ".pdf") -> str:
        """Upload *file_data* to storage and return its ``preview_id``."""
        preview_id = uuid.uuid4().hex
        key = f"{PREVIEW_PREFIX}{preview_id}{suffix}"
        await self._storage.upload_file(key, file_data)
        self._cache[preview_id] = (key, time.monotonic())
        self._cleanup_expired()
        return preview_id

    @asynccontextmanager
    async def get_path(self, preview_id: str):
        """Download the cached file to a temp path and yield it.

        The temp file is deleted when the context exits.  Yields ``None``
        if the preview is expired or not found.
        """
        entry = self._cache.get(preview_id)
        if entry is None:
            yield None
            return

        key, ts = entry
        if time.monotonic() - ts > self._ttl:
            self._evict(preview_id)
            yield None
            return

        tmp_path = await self._storage.download_to_temp(key)
        try:
            yield tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)

    async def get_bytes(self, preview_id: str) -> bytes | None:
        """Download and return raw bytes, or ``None`` if expired / missing."""
        entry = self._cache.get(preview_id)
        if entry is None:
            return None

        key, ts = entry
        if time.monotonic() - ts > self._ttl:
            self._evict(preview_id)
            return None

        tmp_path = await self._storage.download_to_temp(key)
        try:
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    def get_filename(self, preview_id: str) -> str:
        """Return a filename derived from the preview_id."""
        entry = self._cache.get(preview_id)
        if entry is not None:
            key, _ = entry
            return Path(key).name
        return f"{preview_id}.pdf"

    def delete(self, preview_id: str) -> None:
        """Explicitly remove a cached entry from storage."""
        self._evict(preview_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict(self, preview_id: str) -> None:
        entry = self._cache.pop(preview_id, None)
        if entry is not None:
            key, _ = entry
            try:
                self._storage.delete_file(key)
            except Exception:
                logger.warning("Failed to delete cached preview from storage: %s", key)

    def _cleanup_expired(self) -> None:
        now = time.monotonic()
        expired = [pid for pid, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for pid in expired:
            self._evict(pid)
