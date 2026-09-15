"""Sparse index admin adapter — implements ``SparseIndexAdminPort`` for BM25+S3.

Encapsulates all mechanics that were previously inline in
``IngestionService._build_bm25_index``: load-from-S3, BM25Index construction,
save-to-S3, and Redis pub/sub invalidation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from application.ports.sparse_index import ChunkRef
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.bm25_invalidation import publish_bm25_invalidation
from infrastructure.bm25.persistence import load_bm25_index_from_s3, save_bm25_index_to_s3

log = logging.getLogger("default")


class S3SparseIndexAdmin:
    """Implements ``SparseIndexAdminPort`` backed by S3-persisted BM25 index."""

    def __init__(self, file_storage: object) -> None:
        self._file_storage = file_storage

    async def rebuild(self, chunks: Sequence[ChunkRef]) -> None:
        """Full rebuild: replace the entire index with the given chunks."""
        bm25_index = BM25Index(
            [c.text for c in chunks],
            doc_visibility=[c.visibility for c in chunks],
            doc_owner_id=[c.owner_id for c in chunks],
            doc_group_id=[c.group_id for c in chunks],
        )
        await save_bm25_index_to_s3(bm25_index, self._file_storage)
        await publish_bm25_invalidation()
        log.info("BM25: rebuild complete — %d docs", len(chunks))

    async def extend(self, chunks: Sequence[ChunkRef]) -> None:
        """Incremental add: merge new chunks into the existing index."""
        existing = await load_bm25_index_from_s3(self._file_storage)
        if existing is not None:
            all_texts = existing.texts + [c.text for c in chunks]
            all_vis = existing.doc_visibility + [c.visibility for c in chunks]
            all_owners = existing.doc_owner_id + [c.owner_id for c in chunks]
            all_groups = existing.doc_group_id + [c.group_id for c in chunks]
        else:
            all_texts = [c.text for c in chunks]
            all_vis = [c.visibility for c in chunks]
            all_owners = [c.owner_id for c in chunks]
            all_groups = [c.group_id for c in chunks]

        bm25_index = BM25Index(
            all_texts,
            doc_visibility=all_vis,
            doc_owner_id=all_owners,
            doc_group_id=all_groups,
        )
        await save_bm25_index_to_s3(bm25_index, self._file_storage)
        await publish_bm25_invalidation()
        log.info("BM25: extend complete — %d total docs", len(all_texts))
