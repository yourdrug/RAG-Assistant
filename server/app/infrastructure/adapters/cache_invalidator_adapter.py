"""Infrastructure adapter for CacheInvalidatorPort — wraps answer_cache."""

from __future__ import annotations

from infrastructure.ml.answer_cache import invalidate_by_document_ids as _invalidate


class CacheInvalidatorAdapter:
    """Thin wrapper making answer cache invalidation available as an injectable port."""

    async def invalidate_by_document_ids(
        self,
        document_ids: list[int],
        *,
        cache_enabled: bool = True,
    ) -> int:
        return await _invalidate(document_ids, cache_enabled=cache_enabled)
