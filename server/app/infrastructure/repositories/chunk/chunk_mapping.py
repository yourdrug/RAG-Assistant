"""Map stored chunk rows to the repository DTO."""

from domain.repositories.chunk_repository import ChunkSearchResult

from infrastructure.database.models import ChunkModel


def to_chunk_search_result(orm: ChunkModel) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=orm.id,
        document_id=orm.document_id,
        filename=orm.filename,
        content=orm.content,
        chunk_index=orm.chunk_index,
        visibility=orm.visibility,
        doc_domain=orm.doc_domain,
        owner_id=orm.owner_id,
        group_id=orm.group_id,
        edited_at=orm.edited_at,
        edited_by=orm.edited_by,
        manual=orm.manual,
        creation_date=orm.creation_date,
        content_hash=orm.content_hash,
        section=orm.section,
        heading=orm.heading,
        heading_level=orm.heading_level,
        content_type=orm.content_type,
        doc_title=orm.doc_title,
        doc_type=orm.doc_type,
        act_version_id=orm.act_version_id,
        effective_from=orm.effective_from,
        effective_to=orm.effective_to,
        is_current=orm.is_current,
        context_metadata=orm.context_metadata or {},
    )
