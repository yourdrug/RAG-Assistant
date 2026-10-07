"""SQLAlchemy chunk context operations using the caller's session."""

from __future__ import annotations

from domain.repositories.chunk_repository import ChunkSearchResult
from domain.value_objects.chunk_context import TABLE_CONTEXT_MAX_CHUNKS
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel
from infrastructure.repositories.chunk.chunk_mapping import to_chunk_search_result
from infrastructure.repositories.chunk.chunk_search import build_acl_clauses


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
        acl_clauses = build_acl_clauses(
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
        table_id: str,
        exclude_hashes: set[str] | None = None,
        *,
        user,
        limit: int = TABLE_CONTEXT_MAX_CHUNKS,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> list[ChunkSearchResult]:
        """Fetch bounded batches of exactly one table, optionally overlapping rows."""
        if not table_id:
            return []
        if limit <= 0 or limit > TABLE_CONTEXT_MAX_CHUNKS:
            raise ValueError(f"limit must be between 1 and {TABLE_CONTEXT_MAX_CHUNKS}")
        conditions = [
            ChunkModel.document_id == document_id,
            ChunkModel.context_metadata["table_id"].as_string() == table_id,
        ]
        if row_start is not None:
            conditions.append(ChunkModel.context_metadata["table_row_end"].as_integer() >= row_start)
        if row_end is not None:
            conditions.append(ChunkModel.context_metadata["table_row_start"].as_integer() <= row_end)
        if exclude_hashes:
            safe_hashes = {h for h in exclude_hashes if h is not None}
            if safe_hashes:
                conditions.append(~ChunkModel.content_hash.in_(safe_hashes))
        acl_clauses = build_acl_clauses(
            user,
            user.group_ids,
            managed_client_ids=getattr(user, "managed_client_ids", None),
            managed_internal_ids=getattr(user, "managed_internal_ids", None),
            managed_group_ids=getattr(user, "managed_group_ids", None),
        )
        if acl_clauses:
            conditions.append(or_(*acl_clauses))
        stmt = select(ChunkModel).where(and_(*conditions)).order_by(ChunkModel.chunk_index).limit(limit)
        result = await self._session.execute(stmt)
        return [to_chunk_search_result(c) for c in result.scalars().all()]
