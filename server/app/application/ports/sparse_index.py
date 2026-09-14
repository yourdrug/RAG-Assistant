"""Sparse index administration port — batch operations for the ingestion pipeline.

The application layer needs: "reflect these chunks in the sparse index".
The adapter encapsulates all mechanics (S3 load/save, BM25Index construction,
Redis invalidation pub/sub).  Tomorrow the adapter may upsert sparse vectors
into Qdrant — the port contract stays unchanged.

Related: ``BM25IndexPort`` in this package handles point mutations (add/remove/replace)
for manual chunk edits.  ``SparseIndexAdminPort`` handles batch operations for
ingestion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ChunkRef:
    """Minimal chunk reference for sparse index rebuild/extend.

    ACL attributes are intentionally excluded for now — the adapter reads them
    from the chunk metadata when constructing the BM25Index.  When D1 §2
    requires explicit per-doc ACL in the index, this dataclass will be extended
    without breaking the contract.
    """

    text: str
    content_hash: str
    visibility: str = field(default="internal_public")
    owner_id: int | None = None
    group_id: int | None = None


@runtime_checkable
class SparseIndexAdminPort(Protocol):
    """Batch administration of the sparse (BM25) index from the ingestion pipeline."""

    async def rebuild(self, chunks: Sequence[ChunkRef]) -> None:
        """Full rebuild (reset mode): replace the entire index with *chunks*."""

    async def extend(self, chunks: Sequence[ChunkRef]) -> None:
        """Incremental add (append mode): merge *chunks* into the existing index."""
