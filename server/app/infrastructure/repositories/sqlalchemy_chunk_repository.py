"""SQLAlchemy implementation of ChunkRepository — pg_trgm substring search."""

from __future__ import annotations

import logging

from domain.repositories.chunk_repository import ChunkRepository, ChunkSearchResult
from domain.services.access_control import get_visibility_conditions
from domain.value_objects.roles import UserKind
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

    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        assigned_client_ids: list[int],
        limit: int = 20,
        mode: str = "exact",
    ) -> list[ChunkSearchResult]:
        if len(query.strip()) < 3:
            return []

        conditions = get_visibility_conditions(
            UserKind(user["kind"]), user["id"], group_ids, assigned_client_ids, for_list=False
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

        search_clause = ChunkModel.content.ilike(f"%{query}%")

        if mode == "exact":
            stmt = (
                select(
                    ChunkModel.id,
                    ChunkModel.document_id,
                    ChunkModel.filename,
                    ChunkModel.content,
                    ChunkModel.chunk_index,
                    func.similarity(ChunkModel.content, query).label("score"),
                )
                .where(search_clause)
                .where(or_(*acl_clauses) if acl_clauses else text("true"))
                .order_by(text("score DESC"))
                .limit(limit)
            )
        else:
            stmt = (
                select(
                    ChunkModel.id,
                    ChunkModel.document_id,
                    ChunkModel.filename,
                    ChunkModel.content,
                    ChunkModel.chunk_index,
                )
                .where(search_clause)
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
            )
            for row in result.all()
        ]

    async def delete_by_document_id(self, document_id: int) -> None:
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
