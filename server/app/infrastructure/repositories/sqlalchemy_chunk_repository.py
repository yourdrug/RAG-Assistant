"""SQLAlchemy implementation of ChunkRepository — pg_trgm substring search."""

from __future__ import annotations

import logging
import re
from datetime import datetime

from domain.repositories.chunk_repository import ChunkSearchResult, ChunkStats
from domain.services.access_control import get_visibility_conditions
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.owner_match import OwnerMatch
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.search_mode import SearchMode
from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel

log = logging.getLogger("default")


class SQLAlchemyChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _to_chunk_search_result(orm: ChunkModel) -> ChunkSearchResult:
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
        )

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
    ) -> None:
        # Delete existing chunks for this document (re-index)
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))

        if not chunks:
            return

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
            )
            for i, content in enumerate(chunks)
        ]
        self._session.add_all(models)
        await self._session.flush()

    async def get_by_id(self, chunk_id: int) -> ChunkSearchResult | None:
        stmt = select(ChunkModel).where(ChunkModel.id == chunk_id)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        return self._to_chunk_search_result(orm)

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

    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
    ) -> list[ChunkSearchResult]:
        if len(query.strip()) < 3:
            return []

        conditions = get_visibility_conditions(
            UserKind(user["kind"]),
            user["id"],
            group_ids,
            for_list=False,
            user_role=UserRole(user.get("role", "user")),
        )

        acl_clauses = []
        for cond in conditions:
            parts = [ChunkModel.visibility == cond.visibility.value]

            if cond.owner_match == OwnerMatch.SELF.value:
                parts.append(ChunkModel.owner_id == user["id"])

            if cond.group_match:
                parts.append(ChunkModel.group_id.in_(group_ids))

            acl_clauses.append(or_(*parts))

        if document_id is not None:
            acl_clauses.append(ChunkModel.document_id == document_id)

        if mode == SearchMode.EXACT.value:
            escaped_query = re.escape(query)
            stmt = (
                select(
                    ChunkModel.id,
                    ChunkModel.document_id,
                    ChunkModel.filename,
                    ChunkModel.content,
                    ChunkModel.chunk_index,
                    ChunkModel.visibility,
                    ChunkModel.doc_domain,
                    ChunkModel.owner_id,
                    ChunkModel.group_id,
                    ChunkModel.edited_at,
                    ChunkModel.edited_by,
                    ChunkModel.manual,
                    ChunkModel.creation_date,
                )
                .where(text("chunks.content ~* :word_pattern"))
                .where(or_(*acl_clauses) if acl_clauses else text("true"))
                .order_by(ChunkModel.id)
                .limit(limit)
                .params(word_pattern=rf"\y{escaped_query}\y")
            )
        else:
            stmt = (
                select(
                    ChunkModel.id,
                    ChunkModel.document_id,
                    ChunkModel.filename,
                    ChunkModel.content,
                    ChunkModel.chunk_index,
                    ChunkModel.visibility,
                    ChunkModel.doc_domain,
                    ChunkModel.owner_id,
                    ChunkModel.group_id,
                    ChunkModel.edited_at,
                    ChunkModel.edited_by,
                    ChunkModel.manual,
                    ChunkModel.creation_date,
                )
                .where(ChunkModel.content.ilike(f"%{query}%"))
                .where(or_(*acl_clauses) if acl_clauses else text("true"))
                .order_by(ChunkModel.id)
                .limit(limit)
            )

        result = await self._session.execute(stmt)
        return [
            ChunkSearchResult(
                chunk_id=row[0],
                document_id=row[1],
                filename=row[2],
                content=row[3],
                chunk_index=row[4],
                visibility=row[5],
                doc_domain=row[6],
                owner_id=row[7],
                group_id=row[8],
                edited_at=row[9],
                edited_by=row[10],
                manual=row[11],
                creation_date=row[12],
            )
            for row in result.all()
        ]

    async def delete_by_document_id(self, document_id: int) -> None:
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int:
        """Update filename for all chunks belonging to a document. Returns count of updated rows."""
        from sqlalchemy import update as sa_update

        stmt = (
            sa_update(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .values(filename=new_filename)
        )
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
        return [self._to_chunk_search_result(c) for c in chunks], total

    async def find_duplicate_by_hash(
        self, document_id: int, content_hash: str, exclude_chunk_id: int | None = None
    ) -> ChunkSearchResult | None:
        conditions = [ChunkModel.document_id == document_id]
        if exclude_chunk_id is not None:
            conditions.append(ChunkModel.id != exclude_chunk_id)

        stmt = select(ChunkModel).where(*conditions)
        result = await self._session.execute(stmt)
        existing_chunks = result.scalars().all()

        import hashlib

        for chunk in existing_chunks:
            chunk_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]
            if chunk_hash == content_hash:
                return self._to_chunk_search_result(chunk)
        return None

    async def get_document_stats(self, document_id: int) -> ChunkStats:
        stmt = select(
            func.count().label("chunks"),
            func.coalesce(func.sum(func.length(ChunkModel.content)), 0).label("chars"),
        ).where(ChunkModel.document_id == document_id)
        result = await self._session.execute(stmt)
        row = result.one()
        return ChunkStats(total_chunks=row.chunks, total_chars=row.chars)

    async def get_all_contents(self) -> list[str]:
        stmt = select(ChunkModel.content).order_by(ChunkModel.document_id, ChunkModel.chunk_index)
        result = await self._session.execute(stmt)
        return [row[0] for row in result.all()]
