"""Live adapter for ChunkSettingsPort -- reads from global settings at access time."""

from __future__ import annotations

from config import settings


class LiveChunkSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def chunk_size(self) -> int:  # type: ignore[override]
        return settings.chunk_size
