"""Substring retrieval with visibility and pack selection filters."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, func, or_, select, text, true, union_all

from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import get_visibility_conditions
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.enums import SearchMode
from domain.rag.classification import extract_structural_references
from domain.rag.document_outline import DocumentOutline, caption_table_number
from infrastructure.database.models import ChunkModel
from infrastructure.repositories.chunk.pack_selection import pack_selection_clause
from infrastructure.repositories.sql_acl import render_visibility_conditions


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import CompoundSelect, Select
    from sqlalchemy.sql.elements import ColumnElement

    from domain.identity.user_context import UserContext
    from domain.knowledge.pack_metadata import MetadataCondition


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
    ChunkModel.pack_metadata,
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
        pack_metadata=row[13] or {},
        content_hash=row[14],
        section=row[15],
        heading=row[16],
        heading_level=row[17],
        content_type=row[18],
        doc_title=row[19],
        doc_type=row[20],
        context_metadata=row[21] or {},
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


def build_substring_search_stmt(
    query: str,
    user: UserContext,
    mode: str,
    document_id: int | None,
    selection: MetadataCondition | None,
    *,
    limit: int | None = None,
    space_id: int | None = None,
) -> Select[Any]:
    """Share content, document, ACL, space and pack selection predicates across both consumers.

    The space and pack selection narrow the documents considered; access is decided by ACL.
    """
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
    stmt = builder(query, acl_clauses, document_id, limit).where(pack_selection_clause(selection))

    if space_id is not None:
        stmt = stmt.where(ChunkModel.space_id == space_id)

    return stmt


class SQLAlchemyChunkSearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search_substring(  # noqa: PLR0913 - keyword-only search scope options
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
        selection: MetadataCondition | None = None,
        space_id: int | None = None,
        document_ids: Sequence[int] | None = None,
    ) -> list[ChunkRecord]:
        """Search chunks; ``document_ids`` runs one limited probe per document in a single query."""
        if len(query.strip()) < MIN_SUBSTRING_LENGTH:
            return []

        stmt: Select[Any] | CompoundSelect

        if document_ids is None:
            stmt = build_substring_search_stmt(
                query, user, mode, document_id, selection, limit=limit, space_id=space_id
            )
        else:
            if document_id is not None:
                message = "Pass either document_id or document_ids, not both"
                raise ValueError(message)

            scoped = [
                build_substring_search_stmt(
                    query, user, mode, doc_id, selection, limit=limit, space_id=space_id
                )
                for doc_id in dict.fromkeys(document_ids)
            ]

            if not scoped:
                return []

            stmt = scoped[0] if len(scoped) == 1 else union_all(*scoped)

        result = await self._session.execute(stmt)
        return [row_to_search_result(row) for row in result.all()]

    async def search_substring_page(  # noqa: PLR0913 - keyword-only search scope options
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        offset: int = 0,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        selection: MetadataCondition | None = None,
        space_id: int | None = None,
    ) -> tuple[list[ChunkRecord], int]:
        if len(query.strip()) < MIN_SUBSTRING_LENGTH:
            return [], 0

        # Materialize only matching IDs. Count and page use one database snapshot,
        # including empty pages beyond the last match, without buffering all content.
        matches = (
            build_substring_search_stmt(query, user, mode, document_id, selection, space_id=space_id)
            .with_only_columns(ChunkModel.id)
            .order_by(None)
            .cte("matching_chunks")
        )
        totals = select(func.count().label("total")).select_from(matches).cte("match_total")
        page = select(matches.c.id).order_by(matches.c.id).limit(limit).offset(offset).cte("match_page")
        stmt = (
            select(*_SEARCH_COLUMNS, totals.c.total)
            .select_from(totals.outerjoin(page, true()).outerjoin(ChunkModel, ChunkModel.id == page.c.id))
            .order_by(ChunkModel.id)
        )
        result = await self._session.execute(stmt)
        rows = result.all()
        return [row_to_search_result(row) for row in rows if row[0] is not None], rows[0][-1]

    async def get_document_outline(self, document_id: int, *, user: UserContext) -> DocumentOutline:
        """Distinct article numbers and table captions of a document under the search ACL."""
        acl_clauses = build_acl_clauses(
            user,
            user.group_ids,
            user.managed_client_ids,
            user.managed_internal_ids,
            user.managed_group_ids,
        )
        article = ChunkModel.context_metadata["article_number"].as_string()
        stmt = (
            select(article, ChunkModel.heading)
            .distinct()
            .where(ChunkModel.document_id == document_id)
            .where(or_(*acl_clauses) if acl_clauses else text("true"))
            .where(or_(article.is_not(None), ChunkModel.heading.regexp_match(r"^\s*Таблица\s+\d", flags="i")))
        )
        rows = (await self._session.execute(stmt)).all()
        articles = {str(row[0]) for row in rows if row[0]}
        tables = {number for row in rows if (number := caption_table_number(row[1]))}
        return DocumentOutline(articles=tuple(sorted(articles)), tables=tuple(sorted(tables)))
