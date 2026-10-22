"""SQL conditions and result pages for document chunk cursors."""

from __future__ import annotations

from domain.knowledge.chunk import ChunkRecord, encode_cursor
from domain.shared.pagination import CursorPage
from sqlalchemy import and_, or_

from infrastructure.database.models import ChunkModel
from infrastructure.repositories.chunk.chunk_mapping import to_chunk_search_result


def build_cursor_forward_conditions(
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


def build_cursor_backward_conditions(cursor: tuple[int, int]) -> list:
    """Build WHERE conditions for backward cursor pagination."""
    ci, cid = cursor
    return [
        or_(
            ChunkModel.chunk_index < ci,
            and_(ChunkModel.chunk_index == ci, ChunkModel.id < cid),
        )
    ]


def paginate_forward(rows: list, limit: int, cursor: tuple[int, int] | None) -> CursorPage[ChunkRecord]:
    """Build CursorPage from forward-fetched rows."""
    has_extra = len(rows) > limit
    page_rows = rows[:limit]
    next_cur = encode_cursor(page_rows[-1].chunk_index, page_rows[-1].id) if has_extra and page_rows else None
    prev_cur = (
        encode_cursor(page_rows[0].chunk_index, page_rows[0].id) if page_rows and cursor is not None else None
    )
    items = [r if isinstance(r, ChunkRecord) else to_chunk_search_result(r) for r in page_rows]
    return CursorPage(items=items, next_cursor=next_cur, prev_cursor=prev_cur)


def paginate_backward(rows: list, limit: int) -> CursorPage[ChunkRecord]:
    """Build CursorPage from backward-fetched rows."""
    has_extra = len(rows) > limit
    page_rows = rows[:limit]
    page_rows.reverse()
    prev_cur = encode_cursor(page_rows[0].chunk_index, page_rows[0].id) if has_extra and page_rows else None
    next_cur = encode_cursor(page_rows[-1].chunk_index, page_rows[-1].id) if page_rows else None
    items = [r if isinstance(r, ChunkRecord) else to_chunk_search_result(r) for r in page_rows]
    return CursorPage(items=items, next_cursor=next_cur, prev_cursor=prev_cur)
