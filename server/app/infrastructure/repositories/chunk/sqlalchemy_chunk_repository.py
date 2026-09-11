"""SQLAlchemy implementation of ChunkRepository — pg_trgm substring search."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

from domain.repositories.chunk_repository import ChunkSearchResult, ChunkStats
from domain.services import get_visibility_conditions
from domain.utils import encode_cursor
from domain.value_objects.cursor_page import CursorPage
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.owner_match import OwnerMatch
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.search_mode import SearchMode
from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel

log = logging.getLogger("default")

# --- Search column list (shared by exact and fuzzy modes) -------------------

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
)


def _row_to_search_result(row) -> ChunkSearchResult:
    """Convert a search query row tuple to ChunkSearchResult."""
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
    )


# --- ACL clause building (extracted from search_substring) ------------------


def _build_acl_clauses(
    user: dict,
    group_ids: list[int],
    managed_client_ids: list[int] | None = None,
    managed_internal_ids: list[int] | None = None,
    managed_group_ids: list[int] | None = None,
) -> list:
    """Build per-condition ACL clauses from user context.

    Each condition is an AND-group of (visibility + owner/group match).
    The full scope is the OR of all conditions.
    """
    conditions = get_visibility_conditions(
        UserKind(user["kind"]),
        user["id"],
        group_ids,
        for_list=False,
        user_role=UserRole(user.get("role", "user")),
        managed_client_ids=managed_client_ids,
        managed_internal_ids=managed_internal_ids,
        managed_group_ids=managed_group_ids,
    )

    acl_clauses = []
    for cond in conditions:
        parts = [ChunkModel.visibility == cond.visibility.value]

        if cond.owner_match == OwnerMatch.SELF.value:
            parts.append(ChunkModel.owner_id == user["id"])

        if cond.owner_match == OwnerMatch.ASSIGNED.value and cond.owner_ids:
            parts.append(ChunkModel.owner_id.in_(cond.owner_ids))

        if cond.group_match:
            effective_group_ids = cond.group_ids if cond.group_ids is not None else group_ids
            parts.append(ChunkModel.group_id.in_(effective_group_ids))

        acl_clauses.append(and_(*parts))

    return acl_clauses


def _build_exact_search_stmt(query: str, acl_clauses: list, document_id: int | None, limit: int):
    """Build SELECT statement for exact word-boundary search."""
    escaped_query = re.escape(query)
    stmt = (
        select(*_SEARCH_COLUMNS)
        .where(text("chunks.content ~* :word_pattern"))
        .where(or_(*acl_clauses) if acl_clauses else text("true"))
        .order_by(ChunkModel.id)
        .limit(limit)
        .params(word_pattern=rf"\y{escaped_query}\y")
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


# --- Cursor pagination helpers (extracted from list_for_document_cursor) -----


def _build_cursor_forward_conditions(
    cursor: tuple[int, int] | None,
) -> list:
    """Build WHERE conditions for forward cursor pagination."""
    if cursor is None:
        return []
    ci, cid = cursor
    return [
        or_(
            ChunkModel.chunk_index > ci,
            and_(ChunkModel.chunk_index == ci, ChunkModel.id > cid),
        )
    ]


def _build_cursor_backward_conditions(cursor: tuple[int, int]) -> list:
    """Build WHERE conditions for backward cursor pagination."""
    ci, cid = cursor
    return [
        or_(
            ChunkModel.chunk_index < ci,
            and_(ChunkModel.chunk_index == ci, ChunkModel.id < cid),
        )
    ]


def _paginate_forward(
    rows: list, limit: int, cursor: tuple[int, int] | None
) -> CursorPage[ChunkSearchResult]:
    """Build CursorPage from forward-fetched rows."""
    has_extra = len(rows) > limit
    page_rows = rows[:limit]
    next_cur = encode_cursor(page_rows[-1].chunk_index, page_rows[-1].id) if has_extra and page_rows else None
    prev_cur = (
        encode_cursor(page_rows[0].chunk_index, page_rows[0].id) if page_rows and cursor is not None else None
    )
    items = [_row_to_search_result(r) if not isinstance(r, ChunkSearchResult) else r for r in page_rows]
    # Handle both ORM objects and raw tuples
    items = []
    for r in page_rows:
        if isinstance(r, ChunkSearchResult):
            items.append(r)
        else:
            items.append(
                ChunkSearchResult(
                    chunk_id=r.id,
                    document_id=r.document_id,
                    filename=r.filename,
                    content=r.content,
                    chunk_index=r.chunk_index,
                    visibility=r.visibility,
                    doc_domain=r.doc_domain,
                    owner_id=r.owner_id,
                    group_id=r.group_id,
                    edited_at=r.edited_at,
                    edited_by=r.edited_by,
                    manual=r.manual,
                    creation_date=r.creation_date,
                )
            )
    return CursorPage(items=items, next_cursor=next_cur, prev_cursor=prev_cur)


def _paginate_backward(rows: list, limit: int) -> CursorPage[ChunkSearchResult]:
    """Build CursorPage from backward-fetched rows."""
    has_extra = len(rows) > limit
    page_rows = rows[:limit]
    page_rows.reverse()
    prev_cur = encode_cursor(page_rows[0].chunk_index, page_rows[0].id) if has_extra and page_rows else None
    next_cur = encode_cursor(page_rows[0].chunk_index, page_rows[0].id) if page_rows else None
    items = []
    for r in page_rows:
        if isinstance(r, ChunkSearchResult):
            items.append(r)
        else:
            items.append(
                ChunkSearchResult(
                    chunk_id=r.id,
                    document_id=r.document_id,
                    filename=r.filename,
                    content=r.content,
                    chunk_index=r.chunk_index,
                    visibility=r.visibility,
                    doc_domain=r.doc_domain,
                    owner_id=r.owner_id,
                    group_id=r.group_id,
                    edited_at=r.edited_at,
                    edited_by=r.edited_by,
                    manual=r.manual,
                    creation_date=r.creation_date,
                )
            )
    return CursorPage(items=items, next_cursor=next_cur, prev_cursor=prev_cur)


# --- Repository ------------------------------------------------------------


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
            section=orm.section,
            heading=orm.heading,
            heading_level=orm.heading_level,
            content_type=orm.content_type,
            doc_title=orm.doc_title,
            doc_type=orm.doc_type,
        )

    # --- CRUD ----------------------------------------------------------------

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
        domain_metadata: dict | None = None,
        act_version_id: int | None = None,
        effective_from: date | None = None,
        effective_to: date | None = None,
        is_current: bool = True,
        sections: list[str | None] | None = None,
        headings: list[str | None] | None = None,
        heading_levels: list[int | None] | None = None,
        content_types: list[str | None] | None = None,
        doc_titles: list[str | None] | None = None,
        doc_types: list[str | None] | None = None,
    ) -> list[int]:
        """Insert chunks for a document. Replaces existing chunks. Returns chunk IDs."""
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))

        if not chunks:
            return []

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
                domain_metadata=domain_metadata,
                act_version_id=act_version_id,
                effective_from=effective_from,
                effective_to=effective_to,
                is_current=is_current,
                section=sections[i] if sections and i < len(sections) else None,
                heading=headings[i] if headings and i < len(headings) else None,
                heading_level=heading_levels[i] if heading_levels and i < len(heading_levels) else None,
                content_type=content_types[i] if content_types and i < len(content_types) else None,
                doc_title=doc_titles[i] if doc_titles and i < len(doc_titles) else None,
                doc_type=doc_types[i] if doc_types and i < len(doc_types) else None,
            )
            for i, content in enumerate(chunks)
        ]
        self._session.add_all(models)
        await self._session.flush()
        return [m.id for m in models]

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

    async def delete_by_document_id(self, document_id: int) -> None:
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int:
        stmt = update(ChunkModel).where(ChunkModel.document_id == document_id).values(filename=new_filename)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount

    # --- Search -------------------------------------------------------------

    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
        managed_client_ids: list[int] | None = None,
        managed_internal_ids: list[int] | None = None,
        managed_group_ids: list[int] | None = None,
    ) -> list[ChunkSearchResult]:
        if len(query.strip()) < 3:
            return []

        acl_clauses = _build_acl_clauses(
            user,
            group_ids,
            managed_client_ids,
            managed_internal_ids,
            managed_group_ids,
        )

        if mode == SearchMode.EXACT.value:
            stmt = _build_exact_search_stmt(query, acl_clauses, document_id, limit)
        else:
            stmt = _build_fuzzy_search_stmt(query, acl_clauses, document_id, limit)

        result = await self._session.execute(stmt)
        return [_row_to_search_result(row) for row in result.all()]

    # --- List / Query -------------------------------------------------------

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

    async def list_for_document_cursor(
        self,
        document_id: int,
        limit: int = 50,
        cursor: tuple[int, int] | None = None,
        direction: str = "next",
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkSearchResult]:
        conditions = [ChunkModel.document_id == document_id]
        if content_hashes:
            conditions.append(ChunkModel.content_hash.in_(content_hashes))

        if direction == "next":
            conditions.extend(_build_cursor_forward_conditions(cursor))
            stmt = (
                select(ChunkModel)
                .where(*conditions)
                .order_by(ChunkModel.chunk_index, ChunkModel.id)
                .limit(limit + 1)
            )
            result = await self._session.execute(stmt)
            return _paginate_forward(list(result.scalars().all()), limit, cursor)

        # direction == "prev"
        if cursor is None:
            from domain.exceptions import ValidationError

            raise ValidationError("cursor is required when direction=prev")
        conditions.extend(_build_cursor_backward_conditions(cursor))
        stmt = (
            select(ChunkModel)
            .where(*conditions)
            .order_by(ChunkModel.chunk_index.desc(), ChunkModel.id.desc())
            .limit(limit + 1)
        )
        result = await self._session.execute(stmt)
        return _paginate_backward(list(result.scalars().all()), limit)

    async def find_duplicate_by_hash(
        self,
        document_id: int,
        content_hash: str,
        exclude_chunk_id: int | None = None,
    ) -> ChunkSearchResult | None:
        conditions = [
            ChunkModel.document_id == document_id,
            ChunkModel.content_hash == content_hash,
        ]
        if exclude_chunk_id is not None:
            conditions.append(ChunkModel.id != exclude_chunk_id)
        stmt = select(ChunkModel).where(*conditions).limit(1)
        result = await self._session.execute(stmt)
        chunk = result.scalar_one_or_none()
        return self._to_chunk_search_result(chunk) if chunk else None

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

    # --- Neighbor enrichment -------------------------------------------------

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: set[str] | None = None,
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
        stmt = select(ChunkModel).where(and_(*conditions)).order_by(ChunkModel.chunk_index)
        result = await self._session.execute(stmt)
        return [self._to_chunk_search_result(c) for c in result.scalars().all()]

    async def get_table_batches(
        self,
        document_id: int,
        anchor_index: int,
        exclude_hashes: set[str] | None = None,
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
        stmt = select(ChunkModel).where(and_(*conditions)).order_by(ChunkModel.chunk_index)
        result = await self._session.execute(stmt)
        return [self._to_chunk_search_result(c) for c in result.scalars().all()]

    # --- Temporal/versioning -------------------------------------------------

    async def set_current_by_act_version_ids(self, act_version_ids: list[int], is_current: bool) -> int:
        if not act_version_ids:
            return 0
        result = await self._session.execute(
            update(ChunkModel)
            .where(ChunkModel.act_version_id.in_(act_version_ids))
            .values(is_current=is_current)
        )
        await self._session.flush()
        return result.rowcount or 0

    async def update_temporal_by_act_version_id(
        self,
        act_version_id: int,
        effective_from: date | None,
        effective_to: date | None,
    ) -> int:
        result = await self._session.execute(
            update(ChunkModel)
            .where(ChunkModel.act_version_id == act_version_id)
            .values(effective_from=effective_from, effective_to=effective_to)
        )
        await self._session.flush()
        return result.rowcount or 0
