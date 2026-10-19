"""Sparse-index projection for newly ingested chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.ingestion.ports.sparse_index import ChunkRef
from domain.value_objects.visibility import DocumentVisibility


if TYPE_CHECKING:
    from application.ingestion.ports.settings import IngestionSettingsPort
    from application.ingestion.ports.sparse_index import SparseIndexAdminPort
    from domain.entities.raw_document import RawDocument

log = logging.getLogger("default")


class SparseIndexBuilder:
    """Projects chunk metadata into the sparse-index administration port."""

    def __init__(
        self,
        settings: IngestionSettingsPort,
        admin: SparseIndexAdminPort | None,
    ) -> None:
        self._settings = settings
        self._admin = admin

    async def update(self, chunks: list[RawDocument], *, reset: bool) -> None:
        if not self._settings.hybrid_enabled:
            return

        if self._admin is None:
            log.warning("SparseIndexAdminPort not injected — skipping sparse index build")
            return

        chunk_refs = [
            ChunkRef(
                text=chunk.page_content,
                content_hash=chunk.metadata.get("content_hash", ""),
                visibility=chunk.metadata.get("visibility", DocumentVisibility.INTERNAL_PUBLIC.value),
                owner_id=chunk.metadata.get("owner_id"),
                group_id=chunk.metadata.get("group_id"),
            )
            for chunk in chunks
        ]

        if reset:
            await self._admin.rebuild(chunk_refs)
        else:
            await self._admin.extend(chunk_refs)
