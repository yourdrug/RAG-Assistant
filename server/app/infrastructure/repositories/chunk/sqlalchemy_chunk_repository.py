"""Compose specialized chunk repositories over one transactional SQLAlchemy session."""

from __future__ import annotations

from domain.value_objects.chunk_context import TABLE_CONTEXT_MAX_CHUNKS
from collections.abc import AsyncIterator
from datetime import date, datetime
from domain.repositories.chunk_repository import ChunkSearchResult, ChunkStats
from domain.value_objects.cursor_page import CursorPage
from domain.value_objects.doc_domain import DocDomain
from sqlalchemy.ext.asyncio import AsyncSession
from infrastructure.repositories.chunk.chunk_crud import SQLAlchemyChunkCrudRepository
from infrastructure.repositories.chunk.chunk_corpus import SQLAlchemyChunkCorpusRepository
from infrastructure.repositories.chunk.chunk_context import SQLAlchemyChunkContextRepository
from infrastructure.repositories.chunk.chunk_versioning import SQLAlchemyChunkVersioningRepository
from infrastructure.repositories.chunk.chunk_search import SQLAlchemyChunkSearchRepository


class SQLAlchemyChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._crud = SQLAlchemyChunkCrudRepository(session)
        self._corpus = SQLAlchemyChunkCorpusRepository(session)
        self._context = SQLAlchemyChunkContextRepository(session)
        self._versioning = SQLAlchemyChunkVersioningRepository(session)
        self._search = SQLAlchemyChunkSearchRepository(session)

    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[str],
        owner_id: int | None = None,
        group_id: int | None = None,
        doc_domain: str = DocDomain.GENERAL.value,
        content_hashes: list[str] | None = None,
        domain_metadata: dict | None = None,
        act_version_id: int | None = None,
        effective_from: date | None = None,
        effective_to: date | None = None,
        is_current: bool = True,
        sections: list[str | None] | None = None,
        headings: list[str | None] | None = None,
        heading_levels: list[int | None] | None = None,
        content_types: list[str | None] | None = None,
        doc_titles: list[str | None] | None = None,
        doc_types: list[str | None] | None = None,
        context_metadata: list[dict] | None = None,
    ) -> list[int]:
        return await self._crud.bulk_insert(
            document_id=document_id,
            filename=filename,
            visibility=visibility,
            chunks=chunks,
            owner_id=owner_id,
            group_id=group_id,
            doc_domain=doc_domain,
            content_hashes=content_hashes,
            domain_metadata=domain_metadata,
            act_version_id=act_version_id,
            effective_from=effective_from,
            effective_to=effective_to,
            is_current=is_current,
            sections=sections,
            headings=headings,
            heading_levels=heading_levels,
            content_types=content_types,
            doc_titles=doc_titles,
            doc_types=doc_types,
            context_metadata=context_metadata,
        )

    async def get_by_id(self, chunk_id: int) -> ChunkSearchResult | None:
        return await self._crud.get_by_id(chunk_id=chunk_id)

    async def get_max_chunk_index(self, document_id: int) -> int:
        return await self._crud.get_max_chunk_index(document_id=document_id)

    async def update_content(self, chunk_id: int, content: str, edited_at: datetime, edited_by: int) -> None:
        return await self._crud.update_content(
            chunk_id=chunk_id, content=content, edited_at=edited_at, edited_by=edited_by
        )

    async def insert_one(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        filename: str,
        visibility: str,
        doc_domain: str,
        owner_id: int | None = None,
        group_id: int | None = None,
        manual: bool = False,
        content_hash: str | None = None,
        context_metadata: dict | None = None,
        act_version_id: int | None = None,
        effective_from: date | None = None,
        effective_to: date | None = None,
        is_current: bool = True,
    ) -> int:
        return await self._crud.insert_one(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            filename=filename,
            visibility=visibility,
            doc_domain=doc_domain,
            owner_id=owner_id,
            group_id=group_id,
            manual=manual,
            content_hash=content_hash,
            context_metadata=context_metadata,
            act_version_id=act_version_id,
            effective_from=effective_from,
            effective_to=effective_to,
            is_current=is_current,
        )

    async def delete_one(self, chunk_id: int) -> None:
        return await self._crud.delete_one(chunk_id=chunk_id)

    async def delete_by_document_id(self, document_id: int) -> None:
        return await self._crud.delete_by_document_id(document_id=document_id)

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int:
        return await self._crud.update_filename_by_document_id(
            document_id=document_id, new_filename=new_filename
        )

    async def list_for_document(
        self, document_id: int, limit: int = 50, offset: int = 0, content_hashes: list[str] | None = None
    ) -> tuple[list[ChunkSearchResult], int]:
        return await self._crud.list_for_document(
            document_id=document_id, limit=limit, offset=offset, content_hashes=content_hashes
        )

    async def list_for_document_cursor(
        self,
        document_id: int,
        limit: int = 50,
        cursor: tuple[int, int] | None = None,
        direction: str = 'next',
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkSearchResult]:
        return await self._crud.list_for_document_cursor(
            document_id=document_id,
            limit=limit,
            cursor=cursor,
            direction=direction,
            content_hashes=content_hashes,
        )

    async def find_duplicate_by_hash(
        self, document_id: int, content_hash: str, exclude_chunk_id: int | None = None
    ) -> ChunkSearchResult | None:
        return await self._crud.find_duplicate_by_hash(
            document_id=document_id, content_hash=content_hash, exclude_chunk_id=exclude_chunk_id
        )

    async def get_document_stats(self, document_id: int) -> ChunkStats:
        return await self._crud.get_document_stats(document_id=document_id)

    async def get_all_contents(self) -> list[str]:
        return await self._corpus.get_all_contents()

    async def get_all_contents_batches(self, batch_size: int = 5000) -> list[list[str]]:
        return await self._corpus.get_all_contents_batches(batch_size=batch_size)

    def iter_all_contents_with_acl(
        self, batch_size: int = 1000
    ) -> AsyncIterator[tuple[str, str, int | None, int | None]]:
        return self._corpus.iter_all_contents_with_acl(batch_size=batch_size)

    async def get_all_contents_batches_with_acl(
        self, batch_size: int = 5000
    ) -> list[list[tuple[str, str, int | None, int | None]]]:
        return await self._corpus.get_all_contents_batches_with_acl(batch_size=batch_size)

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: set[str] | None = None,
        *,
        user,
    ) -> list[ChunkSearchResult]:
        return await self._context.get_neighbors(
            document_id=document_id,
            center_index=center_index,
            window=window,
            exclude_hashes=exclude_hashes,
            user=user,
        )

    async def get_table_batches(
        self,
        document_id,
        table_id,
        exclude_hashes=None,
        *,
        user,
        limit=TABLE_CONTEXT_MAX_CHUNKS,
        row_start=None,
        row_end=None,
    ):
        return await self._context.get_table_batches(
            document_id,
            table_id,
            exclude_hashes,
            user=user,
            limit=limit,
            row_start=row_start,
            row_end=row_end,
        )

    async def set_current_by_act_version_ids(self, act_version_ids: list[int], is_current: bool) -> int:
        return await self._versioning.set_current_by_act_version_ids(
            act_version_ids=act_version_ids, is_current=is_current
        )

    async def update_temporal_by_act_version_id(
        self, act_version_id: int, effective_from: date | None, effective_to: date | None
    ) -> int:
        return await self._versioning.update_temporal_by_act_version_id(
            act_version_id=act_version_id, effective_from=effective_from, effective_to=effective_to
        )

    async def search_substring(
        self,
        query: str,
        user,
        limit: int = 20,
        mode: str = 'exact',
        document_id: int | None = None,
        as_of_date: date | None = None,
    ) -> list[ChunkSearchResult]:
        return await self._search.search_substring(
            query=query, user=user, limit=limit, mode=mode, document_id=document_id, as_of_date=as_of_date
        )
