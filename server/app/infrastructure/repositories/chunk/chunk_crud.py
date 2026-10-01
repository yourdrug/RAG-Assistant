"""SQLAlchemy chunk crud operations using the caller's session."""

from __future__ import annotations

from datetime import date, datetime

from domain.repositories.chunk_repository import ChunkSearchResult, ChunkStats
from domain.value_objects.cursor_page import CursorPage
from domain.value_objects.doc_domain import DocDomain
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel
from infrastructure.repositories.chunk.chunk_mapping import to_chunk_search_result
from infrastructure.repositories.chunk.chunk_pagination import (
    _build_cursor_backward_conditions,
    _build_cursor_forward_conditions,
    _paginate_backward,
    _paginate_forward,
)


class SQLAlchemyChunkCrudRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
    ) -> list[int]:
        """Insert chunks for a document. Replaces existing chunks. Returns chunk IDs."""
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))

        if not chunks:
            return []

        models = [
            ChunkModel(
                document_id=document_id,
                chunk_index=i,
                content=content,
                filename=filename,
                visibility=visibility,
                doc_domain=doc_domain,
                owner_id=owner_id,
                group_id=group_id,
                content_hash=content_hashes[i] if content_hashes and i < len(content_hashes) else None,
                domain_metadata=domain_metadata,
                act_version_id=act_version_id,
                effective_from=effective_from,
                effective_to=effective_to,
                is_current=is_current,
                section=sections[i] if sections and i < len(sections) else None,
                heading=headings[i] if headings and i < len(headings) else None,
                heading_level=heading_levels[i] if heading_levels and i < len(heading_levels) else None,
                content_type=content_types[i] if content_types and i < len(content_types) else None,
                doc_title=doc_titles[i] if doc_titles and i < len(doc_titles) else None,
                doc_type=doc_types[i] if doc_types and i < len(doc_types) else None,
            )
            for i, content in enumerate(chunks)
        ]
        self._session.add_all(models)
        await self._session.flush()
        return [m.id for m in models]

    async def get_by_id(self, chunk_id: int) -> ChunkSearchResult | None:
        stmt = select(ChunkModel).where(ChunkModel.id == chunk_id)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        return to_chunk_search_result(orm)

    async def get_max_chunk_index(self, document_id: int) -> int:
        stmt = select(func.max(ChunkModel.chunk_index)).where(ChunkModel.document_id == document_id)
        result = await self._session.execute(stmt)
        max_index = result.scalar()
        return max_index if max_index is not None else -1

    async def update_content(
        self,
        chunk_id: int,
        content: str,
        edited_at: datetime,
        edited_by: int,
    ) -> None:
        stmt = select(ChunkModel).where(ChunkModel.id == chunk_id)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"Chunk {chunk_id} not found")
        orm.content = content
        orm.edited_at = edited_at
        orm.edited_by = edited_by
        await self._session.flush()

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
    ) -> int:
        orm = ChunkModel(
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
        )
        self._session.add(orm)
        await self._session.flush()
        return orm.id

    async def delete_one(self, chunk_id: int) -> None:
        stmt = select(ChunkModel).where(ChunkModel.id == chunk_id)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is not None:
            await self._session.delete(orm)
            await self._session.flush()

    async def delete_by_document_id(self, document_id: int) -> None:
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int:
        stmt = update(ChunkModel).where(ChunkModel.document_id == document_id).values(filename=new_filename)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount

    async def list_for_document(
        self,
        document_id: int,
        limit: int = 50,
        offset: int = 0,
        content_hashes: list[str] | None = None,
    ) -> tuple[list[ChunkSearchResult], int]:
        conditions = [ChunkModel.document_id == document_id]
        if content_hashes:
            conditions.append(ChunkModel.content_hash.in_(content_hashes))

        count_stmt = select(func.count()).select_from(ChunkModel).where(*conditions)
        total_result = await self._session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = (
            select(ChunkModel).where(*conditions).order_by(ChunkModel.chunk_index).limit(limit).offset(offset)
        )
        result = await self._session.execute(stmt)
        chunks = result.scalars().all()
        return [to_chunk_search_result(c) for c in chunks], total

    async def list_for_document_cursor(
        self,
        document_id: int,
        limit: int = 50,
        cursor: tuple[int, int] | None = None,
        direction: str = "next",
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkSearchResult]:
        conditions = [ChunkModel.document_id == document_id]
        if content_hashes:
            conditions.append(ChunkModel.content_hash.in_(content_hashes))

        if direction == "next":
            conditions.extend(_build_cursor_forward_conditions(cursor))
            stmt = (
                select(ChunkModel)
                .where(*conditions)
                .order_by(ChunkModel.chunk_index, ChunkModel.id)
                .limit(limit + 1)
            )
            result = await self._session.execute(stmt)
            return _paginate_forward(list(result.scalars().all()), limit, cursor)

        # direction == "prev"
        if cursor is None:
            from domain.exceptions import ValidationError

            raise ValidationError("cursor is required when direction=prev")
        conditions.extend(_build_cursor_backward_conditions(cursor))
        stmt = (
            select(ChunkModel)
            .where(*conditions)
            .order_by(ChunkModel.chunk_index.desc(), ChunkModel.id.desc())
            .limit(limit + 1)
        )
        result = await self._session.execute(stmt)
        return _paginate_backward(list(result.scalars().all()), limit)

    async def find_duplicate_by_hash(
        self,
        document_id: int,
        content_hash: str,
        exclude_chunk_id: int | None = None,
    ) -> ChunkSearchResult | None:
        conditions = [
            ChunkModel.document_id == document_id,
            ChunkModel.content_hash == content_hash,
        ]
        if exclude_chunk_id is not None:
            conditions.append(ChunkModel.id != exclude_chunk_id)
        stmt = select(ChunkModel).where(*conditions).limit(1)
        result = await self._session.execute(stmt)
        chunk = result.scalar_one_or_none()
        return to_chunk_search_result(chunk) if chunk else None

    async def get_document_stats(self, document_id: int) -> ChunkStats:
        stmt = select(
            func.count().label("chunks"),
            func.coalesce(func.sum(func.length(ChunkModel.content)), 0).label("chars"),
        ).where(ChunkModel.document_id == document_id)
        result = await self._session.execute(stmt)
        row = result.one()
        return ChunkStats(total_chunks=row.chunks, total_chars=row.chars)
