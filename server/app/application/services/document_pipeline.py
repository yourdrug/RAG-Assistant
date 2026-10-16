"""Compatibility exports for shared indexing operations."""

from application.indexing import (
    build_outbox_metadata,
    classify_domain,
    enqueue_delete_by_document,
    enrich_chunks_metadata,
    process_chunks,
    process_chunks_in_uow,
    tag_chunks,
    tag_domain,
)


__all__ = [
    "build_outbox_metadata",
    "classify_domain",
    "enqueue_delete_by_document",
    "enrich_chunks_metadata",
    "process_chunks",
    "process_chunks_in_uow",
    "tag_chunks",
    "tag_domain",
]
