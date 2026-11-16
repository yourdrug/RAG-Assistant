"""Read-only view of a chunk's point in the vector index, for admin diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from application.chunks.dto import IndexedPointSnapshot


class ChunkIndexInspectorPort(Protocol):
    async def inspect_point(self, point_id: int) -> IndexedPointSnapshot | None:
        """Return the stored point by ID, or None when it is absent from the index."""
        ...
