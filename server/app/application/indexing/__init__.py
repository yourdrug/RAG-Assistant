"""Public shared indexing operations and results."""

from application.indexing.classification import classify_document_text
from application.indexing.dto import DomainClassification
from application.indexing.pipeline import (
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
    "DomainClassification",
    "build_outbox_metadata",
    "classify_document_text",
    "classify_domain",
    "enqueue_delete_by_document",
    "enrich_chunks_metadata",
    "process_chunks",
    "process_chunks_in_uow",
    "tag_chunks",
    "tag_domain",
]
