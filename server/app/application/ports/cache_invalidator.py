"""CacheInvalidatorPort — invalidates answer cache entries by document IDs."""

from __future__ import annotations

from typing import Protocol


class CacheInvalidatorPort(Protocol):
    """Invalidates cached RAG answers for specific documents."""

    async def invalidate_by_document_ids(
        self,
        document_ids: list[int],
        *,
        cache_enabled: bool = True,
        raise_on_error: bool = False,
    ) -> int: ...
