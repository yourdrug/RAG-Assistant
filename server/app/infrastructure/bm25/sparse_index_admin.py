"""Sparse index admin adapter — implements ``SparseIndexAdminPort`` for BM25+S3.

Encapsulates all mechanics that were previously inline in
``IngestionService._build_bm25_index``: load-from-S3, BM25Index construction,
save-to-S3, and Redis pub/sub invalidation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from application.ingestion.sparse_index_port import ChunkRef
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.bm25_invalidation import publish_bm25_invalidation
from infrastructure.bm25.persistence import save_bm25_index_to_s3

if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory

log = logging.getLogger("default")


class S3SparseIndexAdmin:
    """Implements ``SparseIndexAdminPort`` backed by S3-persisted BM25 index."""

    def __init__(self, file_storage: object, uow_factory: UnitOfWorkFactory) -> None:
        self._file_storage = file_storage
        self._uow_factory = uow_factory

    async def rebuild(self, chunks: Sequence[ChunkRef]) -> None:
        """Full rebuild: replace the entire index with the given chunks."""
        bm25_index = await asyncio.to_thread(
            BM25Index,
            [c.text for c in chunks],
            doc_visibility=[c.visibility for c in chunks],
            doc_owner_id=[c.owner_id for c in chunks],
            doc_group_id=[c.group_id for c in chunks],
        )
        await save_bm25_index_to_s3(bm25_index, self._file_storage)
        await publish_bm25_invalidation()
        log.info("BM25: rebuild complete — %d docs", len(chunks))

    async def extend(self, chunks: Sequence[ChunkRef]) -> None:
        """Refresh from committed SQL so replacements cannot append stale texts.

        The persisted index has no document IDs, so a full corpus snapshot is
        needed to retain unaffected documents and remove replaced chunks safely.
        """
        current = []
        async with self._uow_factory.create(master=True) as uow:
            async for text, visibility, owner_id, group_id in uow.chunks.iter_all_contents_with_acl():
                current.append(ChunkRef(text, "", visibility, owner_id, group_id))
        await self.rebuild(current)
