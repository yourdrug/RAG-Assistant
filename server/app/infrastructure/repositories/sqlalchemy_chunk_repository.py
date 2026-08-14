"""SQLAlchemy implementation of ChunkRepository — pg_trgm substring search."""

from __future__ import annotations

import logging
import re
from datetime import datetime

from domain.repositories.chunk_repository import ChunkRepository, ChunkSearchResult
from domain.services.access_control import get_visibility_conditions
from domain.value_objects.roles import UserKind, UserRole
from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel

log = logging.getLogger("default")


class SQLAlchemyChunkRepository(ChunkRepository):
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
        doc_domain: str = "general",
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
        )

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
        assigned_client_ids: list[int],
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
            assigned_client_ids,
            for_list=False,
            user_role=UserRole(user.get("role", "user")),
        )

        acl_clauses = []
        for cond in conditions:
            parts = [ChunkModel.visibility == cond.visibility.value]

            if cond.owner_match == "self":
                parts.append(ChunkModel.owner_id == user["id"])
            elif cond.owner_match == "assigned":
                parts.append(ChunkModel.owner_id.in_(assigned_client_ids))

            if cond.group_match:
                parts.append(ChunkModel.group_id.in_(group_ids))

            acl_clauses.append(or_(*parts))

        if document_id is not None:
            acl_clauses.append(ChunkModel.document_id == document_id)

        if mode == "exact":
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
                chunk_id=row.id,
                document_id=row.document_id,
                filename=row.filename,
                content=row.content,
                chunk_index=row.chunk_index,
                visibility=row.visibility,
                doc_domain=row.doc_domain,
                owner_id=row.owner_id,
                group_id=row.group_id,
                edited_at=row.edited_at,
                edited_by=row.edited_by,
                manual=row.manual,
                creation_date=row.creation_date,
            )
            for row in result.all()
        ]

    async def delete_by_document_id(self, document_id: int) -> None:
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
