"""SQL conditions and result pages for document chunk cursors."""

from __future__ import annotations

from domain.repositories.chunk_repository import ChunkSearchResult
from domain.utils import encode_cursor
from domain.value_objects.cursor_page import CursorPage
from sqlalchemy import and_, or_

from infrastructure.database.models import ChunkModel


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
