"""Reconstruct retrieved chunk metadata from authoritative PostgreSQL fields."""

from typing import Any

from domain.knowledge.chunk import ChunkRecord, extract_chunk_context
from domain.knowledge.pack_metadata import PACK_METADATA_KEY
from domain.shared.hashing import content_hash


def chunk_result_metadata(chunk: ChunkRecord) -> dict[str, Any]:
    metadata = extract_chunk_context(chunk.context_metadata)

    for key in ("section", "heading", "heading_level", "content_type", "doc_title", "doc_type"):
        value = getattr(chunk, key)

        if value is not None:
            metadata[key] = value

    metadata.update(
        source=chunk.filename,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        content_hash=chunk.content_hash or content_hash(chunk.content),
        visibility=chunk.visibility,
        owner_id=chunk.owner_id,
        group_id=chunk.group_id,
        doc_domain=chunk.doc_domain,
    )

    if chunk.pack_metadata:
        metadata[PACK_METADATA_KEY] = dict(chunk.pack_metadata)

    return metadata
