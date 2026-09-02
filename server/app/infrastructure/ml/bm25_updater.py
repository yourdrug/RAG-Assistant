"""Incremental BM25 index updates outside the full ingestion pipeline.

Provides thin wrappers around BM25Index.add_text / replace_text / remove_text
that operate on the process-wide index via MLClientRegistry.  Used by
ChunkService and DocumentProcessor to keep the sparse index in sync after
manual edits.

All mutations are in-memory only — the periodic rebuild (3:00 UTC) persists
the index to S3, acting as a safety net against drift.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry

log = logging.getLogger("default")


def _find_index_by_hash(idx, old_hash: str) -> int | None:
    """Find the BM25 internal index for a given content hash (O(n) scan)."""
    try:
        return idx.hashes.index(old_hash)
    except ValueError:
        return None


def bm25_add(registry: MLClientRegistry, text: str, text_hash: str | None = None) -> None:
    """Add a new text to the BM25 index."""
    idx = registry.bm25_index()
    if idx is None:
        return
    try:
        idx.add_text(text, text_hash=text_hash)
        log.debug("BM25: added text (hash=%s, n_docs=%d)", text_hash, idx.n_docs)
    except Exception:
        log.exception("BM25: failed to add text")


def bm25_replace(
    registry: MLClientRegistry,
    old_hash: str,
    new_text: str,
    new_hash: str | None = None,
) -> None:
    """Replace text identified by *old_hash* with *new_text*.

    If the old hash is not found in the index (e.g. after a rebuild),
    the new text is appended instead.
    """
    idx = registry.bm25_index()
    if idx is None:
        return
    try:
        pos = _find_index_by_hash(idx, old_hash)
        if pos is not None:
            idx.replace_text(pos, new_text, new_hash=new_hash)
            log.debug("BM25: replaced text at pos %d (n_docs=%d)", pos, idx.n_docs)
        else:
            idx.add_text(new_text, text_hash=new_hash)
            log.debug("BM25: old hash %s not found, appended new text", old_hash)
    except Exception:
        log.exception("BM25: failed to replace text for hash %s", old_hash)


def bm25_remove(registry: MLClientRegistry, old_hash: str) -> None:
    """Remove text identified by *old_hash* from the BM25 index.

    If the hash is not found (already removed or after rebuild), this is
    a no-op.
    """
    idx = registry.bm25_index()
    if idx is None:
        return
    try:
        pos = _find_index_by_hash(idx, old_hash)
        if pos is not None:
            idx.remove_text(pos)
            log.debug("BM25: removed text at pos %d (n_docs=%d)", pos, idx.n_docs)
        else:
            log.debug("BM25: hash %s not found, skip remove", old_hash)
    except Exception:
        log.exception("BM25: failed to remove text for hash %s", old_hash)
