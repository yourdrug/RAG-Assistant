"""Substring retrieval with visibility and temporal filters."""

from __future__ import annotations

import re
from datetime import date

from domain.repositories.chunk_repository import ChunkSearchResult
from domain.services import get_visibility_conditions
from domain.value_objects.owner_match import OwnerMatch
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.search_mode import SearchMode
from sqlalchemy import and_, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ActVersionModel, ChunkModel

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


def _row_to_search_result(row) -> ChunkSearchResult:
    """Convert a search query row tuple to ChunkSearchResult."""
    effective_from = row[15]
    effective_to = row[16]
    return ChunkSearchResult(
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
        user_id,
        group_ids,
        for_list=False,
        user_role=UserRole(user_role_raw),
        managed_client_ids=managed_client_ids,
        managed_internal_ids=managed_internal_ids,
        managed_group_ids=managed_group_ids,
    )

    acl_clauses = []
    for cond in conditions:
        parts = [ChunkModel.visibility == cond.visibility.value]

        if cond.owner_match == OwnerMatch.SELF.value:
            parts.append(ChunkModel.owner_id == user_id)

        if cond.owner_match == OwnerMatch.ASSIGNED.value and cond.owner_ids:
            parts.append(ChunkModel.owner_id.in_(cond.owner_ids))

        if cond.group_match:
            effective_group_ids = cond.group_ids if cond.group_ids is not None else group_ids
            parts.append(ChunkModel.group_id.in_(effective_group_ids))

        acl_clauses.append(and_(*parts))

    return acl_clauses


def _build_exact_search_stmt(query: str, acl_clauses: list, document_id: int | None, limit: int):
    """Build SELECT statement for word-boundary search using GIN trigram index.

    Uses ILIKE '% word %' which leverages the existing pg_trgm GIN index
    (unlike the previous regex ~* which did a full seq scan).

    Fallback if >500K chunks and this becomes slow:
      add tsvector column + GIN index (see migration pattern in docs).
    """
    escaped_query = re.escape(query)
    stmt = (
        select(*_SEARCH_COLUMNS)
        .where(ChunkModel.content.ilike(f"% {escaped_query} %"))
        .where(or_(*acl_clauses) if acl_clauses else text("true"))
        .order_by(ChunkModel.id)
        .limit(limit)
    )
    if document_id is not None:
        stmt = stmt.where(ChunkModel.document_id == document_id)
    return stmt


def _build_fuzzy_search_stmt(query: str, acl_clauses: list, document_id: int | None, limit: int):
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


def _build_temporal_clause(as_of_date: date | None):
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
    ) -> list[ChunkSearchResult]:
        if len(query.strip()) < 3:
            return []

        acl_clauses = build_acl_clauses(
            user,
            user.group_ids,
            user.managed_client_ids,
            user.managed_internal_ids,
            user.managed_group_ids,
        )

        if mode == SearchMode.EXACT.value:
            stmt = _build_exact_search_stmt(query, acl_clauses, document_id, limit)
        else:
            stmt = _build_fuzzy_search_stmt(query, acl_clauses, document_id, limit)

        stmt = stmt.outerjoin(ActVersionModel, ChunkModel.act_version_id == ActVersionModel.id).where(
            _build_temporal_clause(as_of_date)
        )

        result = await self._session.execute(stmt)
        return [_row_to_search_result(row) for row in result.all()]
