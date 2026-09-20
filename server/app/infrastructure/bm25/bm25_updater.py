"""Incremental BM25 index updates outside the full ingestion pipeline.

Provides thin wrappers around BM25Index.add_text / replace_text / remove_text
that operate on the process-wide index via the ML client port. Used by
ChunkService and DocumentProcessor to keep the sparse index in sync after
manual edits.

All mutations are in-memory only — the periodic rebuild (3:00 UTC) persists
the index to S3, acting as a safety net against drift.
"""

from __future__ import annotations

import logging
import threading
import time
from application.ports.ml_clients import MLClientPort
from infrastructure.metrics.metrics import BM25_UPDATE_DURATION

log = logging.getLogger("default")

_bm25_lock = threading.Lock()


def _find_index_by_hash(idx, old_hash: str) -> int | None:
    """Find the BM25 internal index for a given content hash (O(1) via reverse index)."""
    return idx.find_by_hash(old_hash)


def bm25_add(
    registry: MLClientPort,
    text: str,
    text_hash: str | None = None,
    visibility: str | None = None,
    owner_id: int | None = None,
    group_id: int | None = None,
) -> None:
    """Add a new text to the BM25 index."""
    idx = registry.bm25_index()
    if idx is None:
        return
    start = time.perf_counter()
    try:
        with _bm25_lock:
            idx.add_text(
                text,
                text_hash=text_hash,
                visibility=visibility,
                owner_id=owner_id,
                group_id=group_id,
            )
        log.debug("BM25: added text (hash=%s, n_docs=%d)", text_hash, idx.n_docs)
    except Exception:
        log.exception("BM25: failed to add text")
    finally:
        BM25_UPDATE_DURATION.labels(operation="add").observe(time.perf_counter() - start)


def bm25_replace(
    registry: MLClientPort,
    old_hash: str,
    new_text: str,
    new_hash: str | None = None,
    visibility: str | None = None,
    owner_id: int | None = None,
    group_id: int | None = None,
) -> None:
    """Replace text identified by *old_hash* with *new_text*.

    If the old hash is not found in the index (e.g. after a rebuild),
    the new text is appended instead.
    """
    idx = registry.bm25_index()
    if idx is None:
        return
    start = time.perf_counter()
    try:
        with _bm25_lock:
            pos = _find_index_by_hash(idx, old_hash)
            if pos is not None:
                idx.replace_text(
                    pos,
                    new_text,
                    new_hash=new_hash,
                    visibility=visibility,
                    owner_id=owner_id,
                    group_id=group_id,
                )
                log.debug("BM25: replaced text at pos %d (n_docs=%d)", pos, idx.n_docs)
            else:
                idx.add_text(
                    new_text,
                    text_hash=new_hash,
                    visibility=visibility,
                    owner_id=owner_id,
                    group_id=group_id,
                )
                log.debug("BM25: old hash %s not found, appended new text", old_hash)
    except Exception:
        log.exception("BM25: failed to replace text for hash %s", old_hash)
    finally:
        BM25_UPDATE_DURATION.labels(operation="replace").observe(time.perf_counter() - start)


def bm25_remove(registry: MLClientPort, old_hash: str) -> None:
    """Remove text identified by *old_hash* from the BM25 index.

    If the hash is not found (already removed or after rebuild), this is
    a no-op.
    """
    idx = registry.bm25_index()
    if idx is None:
        return
    start = time.perf_counter()
    try:
        with _bm25_lock:
            pos = _find_index_by_hash(idx, old_hash)
            if pos is not None:
                idx.remove_text(pos)
                log.debug("BM25: removed text at pos %d (n_docs=%d)", pos, idx.n_docs)
            else:
                log.debug("BM25: hash %s not found, skip remove", old_hash)
    except Exception:
        log.exception("BM25: failed to remove text for hash %s", old_hash)
    finally:
        BM25_UPDATE_DURATION.labels(operation="remove").observe(time.perf_counter() - start)


class BM25IndexAdapter:
    """Implements BM25IndexPort using the ML client port."""

    def __init__(self, registry: MLClientPort) -> None:
        self._registry = registry

    def remove(self, content_hash: str) -> None:
        bm25_remove(self._registry, content_hash)

    def add(
        self,
        text: str,
        *,
        text_hash: str,
        visibility: str | None = None,
        owner_id: int | None = None,
        group_id: int | None = None,
    ) -> None:
        bm25_add(
            self._registry,
            text,
            text_hash=text_hash,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
        )

    def replace(
        self,
        old_hash: str,
        new_text: str,
        *,
        new_hash: str,
        visibility: str | None = None,
        owner_id: int | None = None,
        group_id: int | None = None,
    ) -> None:
        bm25_replace(
            self._registry,
            old_hash,
            new_text,
            new_hash=new_hash,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
        )
