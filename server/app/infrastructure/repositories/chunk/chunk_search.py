"""Substring retrieval with visibility and temporal filters."""

from __future__ import annotations

import re
from datetime import date

from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.access_control import get_visibility_conditions
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.enums import SearchMode
from sqlalchemy import and_, func, or_, select, text, true
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ActVersionModel, ChunkModel
from infrastructure.repositories.sql_acl import render_visibility_conditions

_SEARCH_COLUMNS = (
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
    ChunkModel.act_version_id,
    ActVersionModel.act_id,
    ChunkModel.effective_from,
    ChunkModel.effective_to,
    ChunkModel.is_current,
    ChunkModel.content_hash,
    ChunkModel.section,
    ChunkModel.heading,
    ChunkModel.heading_level,
    ChunkModel.content_type,
    ChunkModel.doc_title,
    ChunkModel.doc_type,
    ChunkModel.context_metadata,
)


def _row_to_search_result(row) -> ChunkRecord:
    """Convert a search query row tuple to ChunkRecord."""
    effective_from = row[15]
    effective_to = row[16]
    return ChunkRecord(
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
        act_version_id=row[13],
        act_id=row[14],
        effective_from=effective_from,
        effective_to=effective_to,
        is_current=row[17],
        content_hash=row[18],
        section=row[19],
        heading=row[20],
        heading_level=row[21],
        content_type=row[22],
        doc_title=row[23],
        doc_type=row[24],
        context_metadata=row[25] or {},
    )


def build_acl_clauses(
    user,
    group_ids: list[int],
    managed_client_ids: list[int] | None = None,
    managed_internal_ids: list[int] | None = None,
    managed_group_ids: list[int] | None = None,
) -> list:
    """Build per-condition ACL clauses from user context.

    Each condition is an AND-group of (visibility + owner/group match).
    The full scope is the OR of all conditions.
    """
    user_id = user.user_id if hasattr(user, "user_id") else user["id"]
    user_kind = user.user_kind if hasattr(user, "user_kind") else user["kind"]
    user_role_raw = user.user_role if hasattr(user, "user_role") else user.get("role", "user")
    conditions = get_visibility_conditions(
        UserKind(user_kind),
        group_ids,
        for_list=False,
        user_role=UserRole(user_role_raw),
        managed_client_ids=managed_client_ids,
        managed_internal_ids=managed_internal_ids,
        managed_group_ids=managed_group_ids,
    )

    return render_visibility_conditions(
        conditions,
        visibility_column=ChunkModel.visibility,
        owner_column=ChunkModel.owner_id,
        group_column=ChunkModel.group_id,
        user_id=user_id,
        group_ids=group_ids,
    )


def build_exact_search_stmt(query: str, acl_clauses: list, document_id: int | None, limit: int | None):
    """Match a literal phrase at word boundaries, including text edges.

    PostgreSQL pg_trgm supports regex predicates as well as ILIKE.
    """
    pattern = rf"(^|[^[:alnum:]_]){re.escape(query)}($|[^[:alnum:]_])"
    stmt = (
        select(*_SEARCH_COLUMNS)
        .where(ChunkModel.content.regexp_match(pattern, flags="i"))
        .where(or_(*acl_clauses) if acl_clauses else text("true"))
        .order_by(ChunkModel.id)
        .limit(limit)
    )

    if document_id is not None:
        stmt = stmt.where(ChunkModel.document_id == document_id)

    return stmt


def build_fuzzy_search_stmt(query: str, acl_clauses: list, document_id: int | None, limit: int | None):
    """Build SELECT statement for ILIKE fuzzy search."""
    stmt = (
        select(*_SEARCH_COLUMNS)
        .where(ChunkModel.content.ilike(f"%{query}%"))
        .where(or_(*acl_clauses) if acl_clauses else text("true"))
        .order_by(ChunkModel.id)
        .limit(limit)
    )

    if document_id is not None:
        stmt = stmt.where(ChunkModel.document_id == document_id)

    return stmt


def build_temporal_clause(as_of_date: date | None):
    effective_date = as_of_date or date.today()
    return and_(
        or_(ChunkModel.effective_from.is_(None), ChunkModel.effective_from <= effective_date),
        or_(ChunkModel.effective_to.is_(None), ChunkModel.effective_to > effective_date),
        or_(
            ChunkModel.is_current.is_(True),
            ChunkModel.effective_from.is_not(None),
            ChunkModel.effective_to.is_not(None),
        ),
    )


def build_substring_search_stmt(
    query: str,
    user,
    mode: str,
    document_id: int | None,
    as_of_date: date | None,
    limit: int | None = None,
):
    """Share content, document, ACL and temporal predicates across both consumers."""
    acl_clauses = build_acl_clauses(
        user,
        user.group_ids,
        user.managed_client_ids,
        user.managed_internal_ids,
        user.managed_group_ids,
    )
    builder = build_exact_search_stmt if mode == SearchMode.EXACT.value else build_fuzzy_search_stmt
    return (
        builder(query, acl_clauses, document_id, limit)
        .outerjoin(ActVersionModel, ChunkModel.act_version_id == ActVersionModel.id)
        .where(build_temporal_clause(as_of_date))
    )


class SQLAlchemyChunkSearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search_substring(
        self,
        query: str,
        user,
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
        as_of_date: date | None = None,
    ) -> list[ChunkRecord]:
        if len(query.strip()) < 3:
            return []

        stmt = build_substring_search_stmt(query, user, mode, document_id, as_of_date, limit)
        result = await self._session.execute(stmt)
        return [_row_to_search_result(row) for row in result.all()]

    async def search_substring_page(
        self,
        query: str,
        user,
        limit: int = 20,
        offset: int = 0,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        as_of_date: date | None = None,
    ) -> tuple[list[ChunkRecord], int]:
        if len(query.strip()) < 3:
            return [], 0

        # Materialize only matching IDs. Count and page use one database snapshot,
        # including empty pages beyond the last match, without buffering all content.
        matches = (
            build_substring_search_stmt(query, user, mode, document_id, as_of_date)
            .with_only_columns(ChunkModel.id)
            .order_by(None)
            .cte("matching_chunks")
        )
        totals = select(func.count().label("total")).select_from(matches).cte("match_total")
        page = select(matches.c.id).order_by(matches.c.id).limit(limit).offset(offset).cte("match_page")
        stmt = (
            select(*_SEARCH_COLUMNS, totals.c.total)
            .select_from(
                totals.outerjoin(page, true())
                .outerjoin(ChunkModel, ChunkModel.id == page.c.id)
                .outerjoin(ActVersionModel, ChunkModel.act_version_id == ActVersionModel.id)
            )
            .order_by(ChunkModel.id)
        )
        result = await self._session.execute(stmt)
        rows = result.all()
        return [_row_to_search_result(row) for row in rows if row[0] is not None], rows[0][-1]
