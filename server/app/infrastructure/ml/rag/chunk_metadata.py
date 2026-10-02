"""Reconstruct retrieved chunk metadata from authoritative PostgreSQL fields."""

from domain.repositories.chunk_repository import ChunkSearchResult
from domain.utils import content_hash
from domain.value_objects.chunk_context import extract_chunk_context


def chunk_result_metadata(chunk: ChunkSearchResult) -> dict:
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
        act_id=chunk.act_id,
        act_version_id=chunk.act_version_id,
        effective_from=chunk.effective_from.isoformat() if chunk.effective_from else None,
        effective_to=chunk.effective_to.isoformat() if chunk.effective_to else None,
        is_current=chunk.is_current,
    )
    return metadata
