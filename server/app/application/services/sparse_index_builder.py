"""Sparse-index projection for newly ingested chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.ports.sparse_index import ChunkRef, SparseIndexAdminPort

if TYPE_CHECKING:
    from application.ports.ingestion_settings import IngestionSettingsPort

log = logging.getLogger("default")


class SparseIndexBuilder:
    """Projects chunk metadata into the sparse-index administration port."""

    def __init__(
        self,
        settings: "IngestionSettingsPort",
        admin: SparseIndexAdminPort | None,
    ) -> None:
        self._settings = settings
        self._admin = admin

    async def update(self, chunks: list, *, reset: bool) -> None:
        if not self._settings.hybrid_enabled:
            return
        if self._admin is None:
            log.warning("SparseIndexAdminPort not injected — skipping sparse index build")
            return

        chunk_refs = [
            ChunkRef(
                text=chunk.page_content,
                content_hash=chunk.metadata.get("content_hash", ""),
                visibility=chunk.metadata.get("visibility", "internal_public"),
                owner_id=chunk.metadata.get("owner_id"),
                group_id=chunk.metadata.get("group_id"),
            )
            for chunk in chunks
        ]
        if reset:
            await self._admin.rebuild(chunk_refs)
        else:
            await self._admin.extend(chunk_refs)
