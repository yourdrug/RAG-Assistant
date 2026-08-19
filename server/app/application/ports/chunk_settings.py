"""Chunk settings protocol -- abstracts dynamic chunking configuration.

Provides read-only access to chunk-related settings that can be hot-reloaded
via ``/admin/config`` without a process restart.
"""

from __future__ import annotations

from typing import Protocol


class ChunkSettingsPort(Protocol):
    @property
    def chunk_size(self) -> int: ...
