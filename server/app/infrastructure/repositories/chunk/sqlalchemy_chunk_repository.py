"""Compose specialized chunk repositories over one transactional SQLAlchemy session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from domain.knowledge.chunk import TABLE_CONTEXT_MAX_CHUNKS
from domain.knowledge.enums import SearchMode
from domain.knowledge.repositories.chunk import (
    DEFAULT_CHUNK_INSERT_OPTIONS,
    DEFAULT_MANUAL_CHUNK_INSERT_OPTIONS,
)
from infrastructure.repositories.chunk.chunk_context import SQLAlchemyChunkContextRepository
from infrastructure.repositories.chunk.chunk_corpus import SQLAlchemyChunkCorpusRepository
from infrastructure.repositories.chunk.chunk_crud import SQLAlchemyChunkCrudRepository
from infrastructure.repositories.chunk.chunk_pack_metadata import SQLAlchemyChunkPackMetadataRepository
from infrastructure.repositories.chunk.chunk_revisions import SQLAlchemyChunkRevisionRepository
from infrastructure.repositories.chunk.chunk_search import SQLAlchemyChunkSearchRepository
from infrastructure.repositories.chunk.document_nodes import SQLAlchemyDocumentNodeRepository


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from domain.identity.user_context import UserContext
    from domain.knowledge.chunk import ChunkInsert, ChunkRecord, ChunkStats
    from domain.knowledge.document_nodes import DocumentNode
    from domain.knowledge.pack_metadata import MetadataCondition
    from domain.knowledge.repositories.chunk import ChunkInsertOptions, ManualChunkInsertOptions
    from domain.knowledge.revision import RevisionSwitch, SourceVersion, StagedRevision
    from domain.rag.document_outline import DocumentOutline
    from domain.shared.pagination import CursorPage


class SQLAlchemyChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._crud = SQLAlchemyChunkCrudRepository(session)
        self._corpus = SQLAlchemyChunkCorpusRepository(session)
        self._context = SQLAlchemyChunkContextRepository(session)
        self._pack_metadata = SQLAlchemyChunkPackMetadataRepository(session)
        self._search = SQLAlchemyChunkSearchRepository(session)
        self._nodes = SQLAlchemyDocumentNodeRepository(session)
        self._revisions = SQLAlchemyChunkRevisionRepository(session)

    async def stage_revision(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        *,
        nodes: list[DocumentNode],
        options: ChunkInsertOptions = DEFAULT_CHUNK_INSERT_OPTIONS,
        source_version: SourceVersion | None = None,
    ) -> StagedRevision:
        return await self._revisions.stage_revision(
            document_id,
            filename,
            visibility,
            chunks,
            nodes=nodes,
            options=options,
            source_version=source_version,
        )

    async def get_staged_revision(self, document_id: int) -> StagedRevision | None:
        return await self._revisions.get_staged_revision(document_id)

    async def staged_source_versions(self, document_ids: list[int]) -> dict[int, SourceVersion]:
        return await self._revisions.staged_source_versions(document_ids)

    async def get_staged_by_ids(
        self, document_id: int, token: str, chunk_ids: list[int]
    ) -> list[ChunkRecord]:
        return await self._revisions.get_staged_by_ids(document_id, token, chunk_ids)

    async def discard_staged_revision(self, document_id: int) -> list[int]:
        return await self._revisions.discard_staged_revision(document_id)

    async def publish_staged_revision(self, document_id: int, token: str) -> RevisionSwitch | None:
        return await self._revisions.publish_staged_revision(document_id, token)

    async def replace_document_nodes(self, document_id: int, nodes: list[DocumentNode]) -> None:
        await self._nodes.replace_document_nodes(document_id, nodes)

    async def get_parent_context(
        self,
        document_id: int,
        node_id: str,
        *,
        user: UserContext,
        max_chars: int,
    ) -> list[ChunkRecord]:
        return await self._context.get_parent_context(document_id, node_id, user=user, max_chars=max_chars)

    async def get_document_outline(self, document_id: int, *, user: UserContext) -> DocumentOutline:
        return await self._search.get_document_outline(document_id, user=user)

    async def get_ids_by_document_id(self, document_id: int, *, for_update: bool = False) -> list[int]:
        return await self._crud.get_ids_by_document_id(document_id, for_update=for_update)

    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        options: ChunkInsertOptions = DEFAULT_CHUNK_INSERT_OPTIONS,
    ) -> list[int]:
        return await self._crud.bulk_insert(
            document_id=document_id, filename=filename, visibility=visibility, chunks=chunks, options=options
        )

    async def get_by_id(self, chunk_id: int) -> ChunkRecord | None:
        return await self._crud.get_by_id(chunk_id=chunk_id)

    async def get_by_ids(self, document_id: int, chunk_ids: list[int]) -> list[ChunkRecord]:
        return await self._crud.get_by_ids(document_id, chunk_ids)

    async def get_max_chunk_index(self, document_id: int) -> int:
        return await self._crud.get_max_chunk_index(document_id=document_id)

    async def update_content(
        self,
        chunk_id: int,
        content: str,
        content_hash: str,
        edited_at: datetime,
        edited_by: int,
        *,
        embed_text: str | None = None,
    ) -> None:
        return await self._crud.update_content(
            chunk_id=chunk_id,
            content=content,
            content_hash=content_hash,
            edited_at=edited_at,
            edited_by=edited_by,
            embed_text=embed_text,
        )

    async def insert_one(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        filename: str,
        visibility: str,
        *,
        doc_domain: str,
        options: ManualChunkInsertOptions = DEFAULT_MANUAL_CHUNK_INSERT_OPTIONS,
    ) -> int:
        return await self._crud.insert_one(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            filename=filename,
            visibility=visibility,
            doc_domain=doc_domain,
            options=options,
        )

    async def delete_one(self, chunk_id: int) -> None:
        return await self._crud.delete_one(chunk_id=chunk_id)

    async def delete_by_document_id(self, document_id: int) -> None:
        await self._nodes.replace_document_nodes(document_id, [])
        return await self._crud.delete_by_document_id(document_id=document_id)

    async def set_acl(
        self, document_id: int, visibility: str, owner_id: int | None, group_id: int | None
    ) -> None:
        await self._crud.set_acl(document_id, visibility, owner_id, group_id)

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int:
        return await self._crud.update_filename_by_document_id(
            document_id=document_id, new_filename=new_filename
        )

    async def list_for_document(
        self, document_id: int, limit: int = 50, offset: int = 0, content_hashes: list[str] | None = None
    ) -> tuple[list[ChunkRecord], int]:
        return await self._crud.list_for_document(
            document_id=document_id, limit=limit, offset=offset, content_hashes=content_hashes
        )

    async def list_for_document_cursor(
        self,
        document_id: int,
        limit: int = 50,
        cursor: tuple[int, int] | None = None,
        direction: str = "next",
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkRecord]:
        return await self._crud.list_for_document_cursor(
            document_id=document_id,
            limit=limit,
            cursor=cursor,
            direction=direction,
            content_hashes=content_hashes,
        )

    async def find_duplicate_by_hash(
        self, document_id: int, content_hash: str, exclude_chunk_id: int | None = None
    ) -> ChunkRecord | None:
        return await self._crud.find_duplicate_by_hash(
            document_id=document_id, content_hash=content_hash, exclude_chunk_id=exclude_chunk_id
        )

    async def get_document_stats(self, document_id: int) -> ChunkStats:
        return await self._crud.get_document_stats(document_id=document_id)

    def iter_all_contents_with_acl(
        self, batch_size: int = 1000
    ) -> AsyncIterator[tuple[str, str, int | None, int | None]]:
        return self._corpus.iter_all_contents_with_acl(batch_size=batch_size)

    async def list_space_filenames(self, space_id: int) -> dict[int, str]:
        return await self._corpus.list_space_filenames(space_id)

    async def list_document_chunks(self, document_ids: list[int]) -> list[ChunkRecord]:
        return await self._corpus.list_document_chunks(document_ids)

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: set[str] | None = None,
        *,
        user: UserContext,
    ) -> list[ChunkRecord]:
        return await self._context.get_neighbors(
            document_id=document_id,
            center_index=center_index,
            window=window,
            exclude_hashes=exclude_hashes,
            user=user,
        )

    async def get_table_batches(
        self,
        document_id: int,
        table_id: str,
        exclude_hashes: set[str] | None = None,
        *,
        user: UserContext,
        limit: int = TABLE_CONTEXT_MAX_CHUNKS,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> list[ChunkRecord]:
        return await self._context.get_table_batches(
            document_id,
            table_id,
            exclude_hashes,
            user=user,
            limit=limit,
            row_start=row_start,
            row_end=row_end,
        )

    async def set_pack_metadata(self, document_id: int, pack_metadata: dict[str, Any]) -> int:
        return await self._pack_metadata.set_pack_metadata(document_id, pack_metadata)

    async def get_pack_metadata(self, document_id: int) -> dict[str, Any] | None:
        return await self._pack_metadata.get_pack_metadata(document_id)

    async def search_substring(  # noqa: PLR0913 - keyword-only search scope options
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
        selection: MetadataCondition | None = None,
        space_id: int | None = None,
        document_ids: Sequence[int] | None = None,
    ) -> list[ChunkRecord]:
        return await self._search.search_substring(
            query=query,
            user=user,
            limit=limit,
            mode=mode,
            document_id=document_id,
            selection=selection,
            space_id=space_id,
            document_ids=document_ids,
        )

    async def search_substring_page(  # noqa: PLR0913 - keyword-only search scope options
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        offset: int = 0,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        selection: MetadataCondition | None = None,
        space_id: int | None = None,
    ) -> tuple[list[ChunkRecord], int]:
        return await self._search.search_substring_page(
            query=query,
            user=user,
            limit=limit,
            offset=offset,
            mode=mode,
            document_id=document_id,
            selection=selection,
            space_id=space_id,
        )
