"""SQLAlchemy chunk context operations using the caller's session."""

from __future__ import annotations

from domain.repositories.chunk_repository import ChunkSearchResult
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel
from infrastructure.repositories.chunk.chunk_mapping import to_chunk_search_result
from infrastructure.repositories.chunk.chunk_search import _build_acl_clauses


class SQLAlchemyChunkContextRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: set[str] | None = None,
        *,
        user,
    ) -> list[ChunkSearchResult]:
        low = center_index - window
        high = center_index + window
        conditions = [
            ChunkModel.document_id == document_id,
            ChunkModel.chunk_index >= low,
            ChunkModel.chunk_index <= high,
        ]
        if exclude_hashes:
            safe_hashes = {h for h in exclude_hashes if h is not None}
            if safe_hashes:
                conditions.append(~ChunkModel.content_hash.in_(safe_hashes))
        acl_clauses = _build_acl_clauses(
            user,
            user.group_ids,
            managed_client_ids=getattr(user, "managed_client_ids", None),
            managed_internal_ids=getattr(user, "managed_internal_ids", None),
            managed_group_ids=getattr(user, "managed_group_ids", None),
        )
        if acl_clauses:
            conditions.append(or_(*acl_clauses))
        stmt = select(ChunkModel).where(and_(*conditions)).order_by(ChunkModel.chunk_index)
        result = await self._session.execute(stmt)
        return [to_chunk_search_result(c) for c in result.scalars().all()]

    async def get_table_batches(
        self,
        document_id: int,
        anchor_index: int,
        exclude_hashes: set[str] | None = None,
        *,
        user,
    ) -> list[ChunkSearchResult]:
        """Fetch all consecutive table batches starting from anchor_index."""
        conditions = [
            ChunkModel.document_id == document_id,
            ChunkModel.chunk_index >= anchor_index,
        ]
        if exclude_hashes:
            safe_hashes = {h for h in exclude_hashes if h is not None}
            if safe_hashes:
                conditions.append(~ChunkModel.content_hash.in_(safe_hashes))
        acl_clauses = _build_acl_clauses(
            user,
            user.group_ids,
            managed_client_ids=getattr(user, "managed_client_ids", None),
            managed_internal_ids=getattr(user, "managed_internal_ids", None),
            managed_group_ids=getattr(user, "managed_group_ids", None),
        )
        if acl_clauses:
            conditions.append(or_(*acl_clauses))
        stmt = select(ChunkModel).where(and_(*conditions)).order_by(ChunkModel.chunk_index)
        result = await self._session.execute(stmt)
        return [to_chunk_search_result(c) for c in result.scalars().all()]
