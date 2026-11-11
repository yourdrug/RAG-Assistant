"""Substring retrieval with visibility and temporal filters."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import re
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, func, or_, select, text, true

from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import get_visibility_conditions
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.enums import SearchMode
from domain.rag.classification import extract_structural_references
from infrastructure.database.models import ActVersionModel, ChunkModel
from infrastructure.repositories.sql_acl import render_visibility_conditions


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select
    from sqlalchemy.sql.elements import ColumnElement

    from domain.identity.user_context import UserContext


MIN_SUBSTRING_LENGTH = 3


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


def row_to_search_result(row: Sequence[Any]) -> ChunkRecord:
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
    user: UserContext | Mapping[str, Any],
    group_ids: Sequence[int],
    managed_client_ids: Sequence[int] | None = None,
    managed_internal_ids: Sequence[int] | None = None,
    managed_group_ids: Sequence[int] | None = None,
) -> list[ColumnElement[bool]]:
    """Build per-condition ACL clauses from user context.

    Each condition is an AND-group of (visibility + owner/group match).
    The full scope is the OR of all conditions.
    """
    if isinstance(user, Mapping):
        user_id, user_kind, user_role_raw = user["id"], user["kind"], user.get("role", UserRole.USER)
    else:
        user_id, user_kind, user_role_raw = user.user_id, user.user_kind, user.user_role

    conditions = get_visibility_conditions(
        UserKind(user_kind),
        list(group_ids),
        for_list=False,
        user_role=UserRole(user_role_raw),
        managed_client_ids=list(managed_client_ids) if managed_client_ids is not None else None,
        managed_internal_ids=list(managed_internal_ids) if managed_internal_ids is not None else None,
        managed_group_ids=list(managed_group_ids) if managed_group_ids is not None else None,
    )

    return render_visibility_conditions(
        conditions,
        visibility_column=ChunkModel.visibility,
        owner_column=ChunkModel.owner_id,
        group_column=ChunkModel.group_id,
        user_id=user_id,
        group_ids=list(group_ids),
    )


def build_exact_search_stmt(
    query: str, acl_clauses: list[ColumnElement[bool]], document_id: int | None, limit: int | None
) -> Select[Any]:
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


def build_provision_search_stmt(
    query: str, acl_clauses: list[ColumnElement[bool]], document_id: int | None, limit: int | None
) -> Select[Any]:
    """Locate a named provision ("Статья 5", "Пункт 3") by structural metadata.

    Chunks without stored structural metadata fall back to a heading-line match.
    A query without a structural reference degrades to the literal exact match.
    """
    references = extract_structural_references(query)

    if not references:
        return build_exact_search_stmt(query, acl_clauses, document_id, limit)

    reference = references[0]
    field = ChunkModel.context_metadata[reference.kind + "_number"].as_string()
    number = re.escape(reference.value)
    marker = {"article": "Статья", "chapter": "Глава", "section": "Раздел", "clause": "Параграф"}.get(
        reference.kind
    )
    fallback = (
        rf"(^|[\n\r])[ \t]*{marker}[ \t]+{number}([.)[:space:]]|$)"
        if marker
        else rf"(^|[\n\r])[ \t]*{number}[.)][[:space:]]"
    )
    predicate = or_(
        field == reference.value,
        and_(field.is_(None), ChunkModel.content.regexp_match(fallback, flags="i")),
    )
    stmt = (
        select(*_SEARCH_COLUMNS)
        .where(predicate)
        .where(or_(*acl_clauses) if acl_clauses else text("true"))
        .order_by(ChunkModel.chunk_index, ChunkModel.id)
        .limit(limit)
    )

    if document_id is not None:
        stmt = stmt.where(ChunkModel.document_id == document_id)

    return stmt


def build_fuzzy_search_stmt(
    query: str, acl_clauses: list[ColumnElement[bool]], document_id: int | None, limit: int | None
) -> Select[Any]:
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


def build_temporal_clause(as_of_date: date | None) -> ColumnElement[bool]:
    effective_date = as_of_date or datetime.now().astimezone().date()
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
    user: UserContext,
    mode: str,
    document_id: int | None,
    as_of_date: date | None,
    *,
    limit: int | None = None,
) -> Select[Any]:
    """Share content, document, ACL and temporal predicates across both consumers."""
    acl_clauses = build_acl_clauses(
        user,
        user.group_ids,
        user.managed_client_ids,
        user.managed_internal_ids,
        user.managed_group_ids,
    )
    builders = {
        SearchMode.EXACT.value: build_exact_search_stmt,
        SearchMode.PROVISION.value: build_provision_search_stmt,
    }
    builder = builders.get(mode, build_fuzzy_search_stmt)
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
        user: UserContext,
        *,
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
        as_of_date: date | None = None,
    ) -> list[ChunkRecord]:
        if len(query.strip()) < MIN_SUBSTRING_LENGTH:
            return []

        stmt = build_substring_search_stmt(query, user, mode, document_id, as_of_date, limit=limit)
        result = await self._session.execute(stmt)
        return [row_to_search_result(row) for row in result.all()]

    async def search_substring_page(
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        offset: int = 0,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        as_of_date: date | None = None,
    ) -> tuple[list[ChunkRecord], int]:
        if len(query.strip()) < MIN_SUBSTRING_LENGTH:
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
        return [row_to_search_result(row) for row in rows if row[0] is not None], rows[0][-1]
