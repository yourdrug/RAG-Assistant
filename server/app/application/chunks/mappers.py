"""Chunk DTO mapping -- single source of truth for ChunkRecord -> ChunkItemDTO."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.chunks.dto import ChunkItemDTO


if TYPE_CHECKING:
    from domain.knowledge.chunk import ChunkRecord


def to_chunk_dto(c: ChunkRecord) -> ChunkItemDTO:
    """Map a ChunkRecord (or compatible) to a ChunkItemDTO."""
    return ChunkItemDTO(
        id=c.chunk_id,
        document_id=c.document_id,
        chunk_index=c.chunk_index,
        content=c.content,
        filename=c.filename,
        visibility=c.visibility,
        doc_domain=c.doc_domain,
        owner_id=c.owner_id,
        group_id=c.group_id,
        edited_at=c.edited_at.isoformat() if c.edited_at else None,
        edited_by=c.edited_by,
        manual=c.manual,
        creation_date=c.creation_date.isoformat() if c.creation_date else None,
        content_hash=c.content_hash,
        section=c.section,
        heading=c.heading,
        heading_level=c.heading_level,
        content_type=c.content_type,
        doc_title=c.doc_title,
        doc_type=c.doc_type,
    )
