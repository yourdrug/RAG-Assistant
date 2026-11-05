"""Tests for cursor-based pagination (domain utils + service layer)."""

from __future__ import annotations

import base64
from pathlib import Path
import sys

import pytest

from domain.identity.roles import UserKind, UserRole
from domain.shared.exceptions import EntityNotFoundError


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from typing import TYPE_CHECKING

from application.chunks.dto import ChunkCursorQuery
from application.chunks.queries import ChunkQueryService
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from domain.shared.exceptions import ValidationError
from domain.shared.pagination import decode_cursor, encode_cursor
from tests.fakes import FakeUnitOfWork, FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from tests.fakes import FakeChunkRepository


# ---------------------------------------------------------------------------
# Domain utils — encode / decode
# ---------------------------------------------------------------------------


class TestCursorEncoding:
    def test_round_trip_basic(self) -> None:
        assert decode_cursor(encode_cursor(0, 1)) == (0, 1)

    def test_round_trip_large_ids(self) -> None:
        assert decode_cursor(encode_cursor(999999, 2**31)) == (999999, 2**31)

    def test_round_trip_chunk_index_zero(self) -> None:
        ci, cid = decode_cursor(encode_cursor(0, 42))
        assert ci == 0
        assert cid == 42

    def test_decode_garbage_raises(self) -> None:
        with pytest.raises(ValidationError, match="Invalid cursor"):
            decode_cursor("!!!not-base64!!!")

    def test_decode_wrong_structure_raises(self) -> None:

        bad = base64.urlsafe_b64encode(b"only-one-part").decode().rstrip("=")
        with pytest.raises(ValidationError, match="Invalid cursor"):
            decode_cursor(bad)

    def test_decode_empty_raises(self) -> None:
        with pytest.raises(ValidationError, match="Invalid cursor"):
            decode_cursor("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(doc_id: int = 1) -> Document:
    """Public document ACL agrees with the seeded chunks."""
    return Document(id=doc_id, visibility=DocumentVisibility.INTERNAL_PUBLIC)


def _seed_chunks(fake_chunks: FakeChunkRepository, document_id: int, count: int) -> None:
    """Insert ``count`` chunks into the fake repo, auto-assigning ids and chunk_index."""
    for i in range(count):
        fake_chunks._chunks.append(
            {
                "id": fake_chunks._next_id,
                "document_id": document_id,
                "content": f"chunk content {i}",
                "chunk_index": i,
                "filename": "test.pdf",
                "visibility": "internal_public",
                "content_hash": None,
            }
        )
        fake_chunks._next_id += 1


# ---------------------------------------------------------------------------
# Service — list_chunks_cursor (fakes)
# ---------------------------------------------------------------------------


class TestChunkServiceCursor:
    @pytest.fixture
    def uow(self) -> FakeUnitOfWork:
        uow = FakeUnitOfWork()
        uow.fake_documents._documents[1] = make_doc(1)
        return uow

    @pytest.fixture
    def factory(self, uow: FakeUnitOfWork) -> FakeUnitOfWorkFactory:
        return FakeUnitOfWorkFactory(uow=uow)

    @pytest.fixture
    def service(self, factory: FakeUnitOfWorkFactory) -> ChunkQueryService:
        return ChunkQueryService(factory)

    @pytest.mark.asyncio
    async def test_first_page_prev_cursor_is_none(
        self, service: ChunkQueryService, uow: FakeUnitOfWork
    ) -> None:
        _seed_chunks(uow.fake_chunks, 1, 3)

        result = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=50),
        )

        assert result.prev_cursor is None
        assert result.next_cursor is None
        assert len(result.items) == 3

    @pytest.mark.asyncio
    async def test_last_page_next_cursor_is_none(
        self, service: ChunkQueryService, uow: FakeUnitOfWork
    ) -> None:
        _seed_chunks(uow.fake_chunks, 1, 5)

        result = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=5),
        )

        assert result.next_cursor is None
        assert result.prev_cursor is None

    @pytest.mark.asyncio
    async def test_direction_prev_without_cursor_raises(self, service: ChunkQueryService) -> None:
        with pytest.raises(ValidationError, match="cursor is required"):
            await service.list_chunks_cursor(
                document_id=1,
                user_id=10,
                user_kind=UserKind.INTERNAL,
                user_role=UserRole.ADMIN,
                query=ChunkCursorQuery(direction="prev"),
            )

    @pytest.mark.asyncio
    async def test_document_not_found_raises(self, service: ChunkQueryService) -> None:

        with pytest.raises(EntityNotFoundError):
            await service.list_chunks_cursor(
                document_id=999,
                user_id=10,
                user_kind=UserKind.INTERNAL,
                user_role=UserRole.ADMIN,
            )

    @pytest.mark.asyncio
    async def test_empty_document_returns_empty_page(self, service: ChunkQueryService) -> None:
        result = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
        )

        assert result.items == []
        assert result.next_cursor is None
        assert result.prev_cursor is None

    @pytest.mark.asyncio
    async def test_content_hashes_forwarded(self, service: ChunkQueryService, uow: FakeUnitOfWork) -> None:
        _seed_chunks(uow.fake_chunks, 1, 3)
        uow.fake_chunks._chunks[0]["content_hash"] = "abc123"

        result = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(content_hashes=["abc123"]),
        )

        assert len(result.items) == 1
        assert result.items[0].content_hash == "abc123"

    @pytest.mark.asyncio
    async def test_pagination_forward(self, service: ChunkQueryService, uow: FakeUnitOfWork) -> None:
        _seed_chunks(uow.fake_chunks, 1, 5)

        page1 = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=2),
        )
        assert len(page1.items) == 2
        assert page1.next_cursor is not None
        assert page1.prev_cursor is None

        page2 = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=2, cursor=page1.next_cursor, direction="next"),
        )
        assert len(page2.items) == 2
        assert page2.next_cursor is not None
        assert page2.prev_cursor is not None

        page3 = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=2, cursor=page2.next_cursor, direction="next"),
        )
        assert len(page3.items) == 1
        assert page3.next_cursor is None

    @pytest.mark.asyncio
    async def test_pagination_backward(self, service: ChunkQueryService, uow: FakeUnitOfWork) -> None:
        _seed_chunks(uow.fake_chunks, 1, 5)

        # navigate forward to get a cursor
        page1 = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=2),
        )
        page2 = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=2, cursor=page1.next_cursor, direction="next"),
        )

        # go back
        back = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=2, cursor=page2.prev_cursor, direction="prev"),
        )
        assert len(back.items) == 2
        assert back.items[0].chunk_index == page1.items[0].chunk_index
        assert back.items[1].chunk_index == page1.items[1].chunk_index

    @pytest.mark.asyncio
    async def test_ordering_is_chunk_index_asc(self, service: ChunkQueryService, uow: FakeUnitOfWork) -> None:
        _seed_chunks(uow.fake_chunks, 1, 10)

        result = await service.list_chunks_cursor(
            document_id=1,
            user_id=10,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            query=ChunkCursorQuery(limit=100),
        )

        indices = [c.chunk_index for c in result.items]
        assert indices == sorted(indices)
