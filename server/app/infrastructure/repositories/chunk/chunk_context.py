"""SQLAlchemy chunk context operations using the caller's session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import and_, or_, select

from domain.knowledge.chunk import TABLE_CONTEXT_MAX_CHUNKS
from domain.knowledge.document_nodes import (
    NODE_SEPARATOR,
    PARENT_NODE_ID_KEY,
    DocumentNode,
    in_scope,
    measured_chain,
    node_ancestry,
    select_parent,
    within_chunk_limit,
)
from infrastructure.database.models import ChunkModel, DocumentNodeModel
from infrastructure.repositories.chunk.chunk_mapping import to_chunk_search_result
from infrastructure.repositories.chunk.chunk_search import build_acl_clauses


if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession

    from domain.identity.user_context import UserContext
    from domain.knowledge.chunk import ChunkRecord


# A parent with more chunks is never read partially; expansion falls back to neighbours.
MAX_PARENT_CHUNKS = 64


def search_acl(user: UserContext) -> list[ColumnElement[bool]]:
    return build_acl_clauses(
        user,
        user.group_ids,
        managed_client_ids=getattr(user, "managed_client_ids", None),
        managed_internal_ids=getattr(user, "managed_internal_ids", None),
        managed_group_ids=getattr(user, "managed_group_ids", None),
    )


class SQLAlchemyChunkContextRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: AbstractSet[str | None] | None = None,
        *,
        user: UserContext,
    ) -> list[ChunkRecord]:
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
        exclude_hashes: AbstractSet[str | None] | None = None,
        *,
        user: UserContext,
        limit: int = TABLE_CONTEXT_MAX_CHUNKS,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> list[ChunkRecord]:
        """Fetch bounded batches of exactly one table, optionally overlapping rows."""
        if not table_id:
            return []

        if limit <= 0 or limit > TABLE_CONTEXT_MAX_CHUNKS:
            msg = f"limit must be between 1 and {TABLE_CONTEXT_MAX_CHUNKS}"
            raise ValueError(msg)

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

    async def get_parent_context(
        self,
        document_id: int,
        node_id: str,
        *,
        user: UserContext,
        act_version_id: int | None,
        max_chars: int,
    ) -> list[ChunkRecord]:
        """Chunks of the largest fitting ancestor scope, read only through the search ACL.

        Nodes are visible only when the user can read a chunk of the same document and edition
        (``for_list=False``: ADMIN does not reach CLIENT_PRIVATE). Empty when no scope fits.
        """
        if not node_id or max_chars <= 0:
            return []

        acl_clauses = search_acl(user)
        visible_chunk = [
            ChunkModel.document_id == document_id,
            ChunkModel.act_version_id.is_not_distinct_from(act_version_id),
        ]

        if acl_clauses:
            visible_chunk.append(or_(*acl_clauses))

        chain = node_ancestry(node_id)
        stmt = select(DocumentNodeModel).where(
            DocumentNodeModel.document_id == document_id,
            DocumentNodeModel.act_version_id.is_not_distinct_from(act_version_id),
            DocumentNodeModel.node_id.in_(chain),
            select(ChunkModel.id).where(*visible_chunk).exists(),
        )
        rows = {row.node_id: row for row in (await self._session.execute(stmt)).scalars().all()}
        nodes = []

        # Stop at a missing ancestor: a gap means stale or foreign scopes.
        for candidate in chain:
            row = rows.get(candidate)

            if row is None:
                break

            nodes.append(
                DocumentNode(
                    node_id=row.node_id,
                    parent_id=row.parent_id,
                    kind=row.kind,
                    label=row.label,
                    number=row.number,
                    depth=row.depth,
                    chunk_start=row.chunk_start,
                    chunk_end=row.chunk_end,
                    pages=tuple(row.pages or ()),
                    char_count=row.char_count,
                    text_hash=row.text_hash,
                )
            )

        # Only scopes readable in full; the largest one bounds a single read.
        candidates = within_chunk_limit(nodes, MAX_PARENT_CHUNKS)

        if not candidates:
            return []

        top = candidates[-1]
        scope = ChunkModel.context_metadata[PARENT_NODE_ID_KEY].as_string()
        chunk_stmt = (
            select(ChunkModel)
            .where(
                *visible_chunk,
                ChunkModel.chunk_index >= top.chunk_start,
                ChunkModel.chunk_index <= top.chunk_end,
                or_(
                    scope == top.node_id,
                    scope.startswith(top.node_id + NODE_SEPARATOR, autoescape=True),
                ),
            )
            .order_by(ChunkModel.chunk_index)
            .limit(MAX_PARENT_CHUNKS)
        )
        records = [to_chunk_search_result(c) for c in (await self._session.execute(chunk_stmt)).scalars()]
        parents = [(record.context_metadata.get(PARENT_NODE_ID_KEY), record.content) for record in records]
        # Stored char_count is an indexing-time value; manual edits change the real size.
        parent = select_parent(measured_chain(candidates, parents), max_chars)

        if parent is None:
            return []

        return [
            record
            for record, (parent_id, _) in zip(records, parents, strict=True)
            if in_scope(parent.node_id, parent_id)
        ]
