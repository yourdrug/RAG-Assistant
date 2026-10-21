"""SQLAlchemy chunk crud operations using the caller's session."""

from __future__ import annotations

from datetime import date, datetime

from domain.knowledge.chunk import ChunkInsert, ChunkRecord, ChunkStats
from domain.shared.exceptions import BusinessRuleViolationError
from domain.shared.pagination import CursorPage
from domain.knowledge.enums import DocDomain
from domain.knowledge.chunk import extract_chunk_context
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel, DocumentModel
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

    async def get_ids_by_document_id(self, document_id: int, *, for_update: bool = False) -> list[int]:
        if for_update:
            # Lock the parent even when it has no chunks, serializing replacements.
            await self._session.execute(
                select(DocumentModel.id).where(DocumentModel.id == document_id).with_for_update()
            )
        result = await self._session.execute(
            select(ChunkModel.id).where(ChunkModel.document_id == document_id)
        )
        return list(result.scalars().all())

    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        owner_id: int | None = None,
        group_id: int | None = None,
        doc_domain: str = DocDomain.GENERAL.value,
        domain_metadata: dict | None = None,
        act_version_id: int | None = None,
        effective_from: date | None = None,
        effective_to: date | None = None,
        is_current: bool = True,
    ) -> list[int]:
        """Insert chunks for a document. Replaces existing chunks. Returns chunk IDs."""
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))

        if not chunks:
            return []

        models = [
            ChunkModel(
                document_id=document_id,
                chunk_index=i,
                content=chunk.content,
                filename=filename,
                visibility=visibility,
                doc_domain=doc_domain,
                owner_id=owner_id,
                group_id=group_id,
                content_hash=chunk.content_hash,
                domain_metadata=domain_metadata,
                act_version_id=act_version_id,
                effective_from=effective_from,
                effective_to=effective_to,
                is_current=is_current,
                section=chunk.section,
                heading=chunk.heading,
                heading_level=chunk.heading_level,
                content_type=chunk.content_type,
                doc_title=chunk.doc_title,
                doc_type=chunk.doc_type,
                context_metadata=extract_chunk_context(chunk.context_metadata),
            )
            for i, chunk in enumerate(chunks)
        ]
        self._session.add_all(models)
        await self._session.flush()
        return [m.id for m in models]

    async def get_by_id(self, chunk_id: int) -> ChunkRecord | None:
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
        content_hash: str,
        edited_at: datetime,
        edited_by: int,
    ) -> None:
        stmt = select(ChunkModel).where(ChunkModel.id == chunk_id)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"Chunk {chunk_id} not found")
        orm.content = content
        orm.content_hash = content_hash
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
        context_metadata: dict | None = None,
        act_version_id: int | None = None,
        effective_from: date | None = None,
        effective_to: date | None = None,
        is_current: bool = True,
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
            context_metadata=extract_chunk_context(context_metadata or {}),
            act_version_id=act_version_id,
            effective_from=effective_from,
            effective_to=effective_to,
            is_current=is_current,
        )
        self._session.add(orm)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            cause = exc.orig.__cause__ if exc.orig is not None else None
            diag = getattr(exc.orig, "diag", None)
            constraint = getattr(cause, "constraint_name", None) or getattr(diag, "constraint_name", None)

            if constraint == "ux_chunks_document_index":
                msg = "Chunk index already exists for this document; retry adding the chunk"
                raise BusinessRuleViolationError(msg) from exc

            raise
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
    ) -> tuple[list[ChunkRecord], int]:
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
    ) -> CursorPage[ChunkRecord]:
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
            from domain.shared.exceptions import ValidationError

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
    ) -> ChunkRecord | None:
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
