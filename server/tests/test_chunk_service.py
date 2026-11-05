"""Tests for chunk mutations and manual-document creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.chunks.mutations import ChunkMutationService
from application.documents import ManualDocumentService
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import compute_owner_and_group
from domain.knowledge.chunk import ChunkRecord, ChunkStats
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility, SourceType
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError, ValidationError
from domain.shared.hashing import content_hash


if TYPE_CHECKING:
    from application.chunks import ChunkSettingsPort
    from application.ports.cache_invalidator import CacheInvalidatorPort


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeChunkSettings:
    chunk_size: int = 550
    chunk_overlap: int = 50
    legal_chunk_size: int = 550
    legal_chunk_overlap: int = 50


def _make_doc(**overrides: Any) -> Document:
    defaults: dict[str, Any] = {
        "id": 1,
        "filename": "test.pdf",
        "visibility": DocumentVisibility.INTERNAL_PUBLIC,
        "owner_id": None,
        "status": DocumentStatus.DONE,
        "source_type": SourceType.FILE.value,
    }
    defaults.update(overrides)
    return Document(**defaults)


def _make_chunk(**overrides: Any) -> ChunkRecord:
    defaults: dict[str, Any] = {
        "chunk_id": 10,
        "document_id": 1,
        "chunk_index": 0,
        "content": "original chunk content " * 10,
        "filename": "test.pdf",
        "visibility": "internal_public",
        "doc_domain": "general",
        "owner_id": None,
        "group_id": None,
        "edited_at": None,
        "edited_by": None,
        "manual": False,
        "creation_date": None,
        "content_hash": "hash_original",
        "section": None,
        "heading": None,
        "heading_level": None,
        "content_type": None,
        "doc_title": None,
        "doc_type": None,
    }
    defaults.update(overrides)
    return ChunkRecord(**defaults)


def _make_uow(doc: Document | None = None, chunk: ChunkRecord | None = None, max_index: int = 0) -> AsyncMock:
    """Build a mock UoW with document + chunk repos."""
    if doc is None:
        doc = _make_doc()

    if chunk is None:
        chunk = _make_chunk()

    uow = AsyncMock()
    uow.documents.get_by_id = AsyncMock(return_value=doc)
    uow.chunks.get_by_id = AsyncMock(return_value=chunk)
    uow.chunks.get_max_chunk_index = AsyncMock(return_value=max_index)
    uow.chunks.find_duplicate_by_hash = AsyncMock(return_value=None)
    uow.chunks.get_document_stats = AsyncMock(return_value=ChunkStats(total_chunks=5, total_chars=1234))
    uow.chunks.insert_one = AsyncMock(return_value=20)
    uow.chunks.update_content = AsyncMock()
    uow.chunks.delete_one = AsyncMock()
    uow.groups.get_user_group_ids = AsyncMock(return_value=[])
    uow.documents.set_has_manual_edits = AsyncMock()
    uow.documents.update_chunk_stats = AsyncMock()
    uow.vector_outbox.enqueue = AsyncMock()
    uow.act_versions.get_by_document_id = AsyncMock(return_value=None)
    return uow


def _make_service(
    uow: AsyncMock,
    *,
    chunk_settings: ChunkSettingsPort | None = None,
    cache_invalidator: CacheInvalidatorPort | None = None,
) -> ChunkMutationService:
    factory = MagicMock()
    factory.create.return_value = AsyncMock(
        __aenter__=AsyncMock(return_value=uow),
        __aexit__=AsyncMock(return_value=False),
    )

    return ChunkMutationService(
        factory,
        chunk_settings if chunk_settings is not None else FakeChunkSettings(),
        cache_invalidator=cache_invalidator,
    )


def make_manual_service(uow: AsyncMock) -> ManualDocumentService:
    factory = MagicMock()
    factory.create.return_value = AsyncMock(
        __aenter__=AsyncMock(return_value=uow),
        __aexit__=AsyncMock(return_value=False),
    )
    return ManualDocumentService(factory)


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


class TestChunkServiceValidation:
    """Tests for ChunkMutationService validation helpers."""

    def test_validate_chunk_content_empty(self) -> None:
        svc = _make_service(AsyncMock())
        with pytest.raises(ValidationError, match="cannot be empty"):
            svc.validate_chunk_content("")
        with pytest.raises(ValidationError, match="cannot be empty"):
            svc.validate_chunk_content("   ")

    def test_validate_chunk_content_too_short(self) -> None:
        svc = _make_service(AsyncMock())
        with pytest.raises(ValidationError, match="too short"):
            svc.validate_chunk_content("ab")

    def test_validate_chunk_content_too_long(self) -> None:
        svc = _make_service(AsyncMock())
        with pytest.raises(ValidationError, match="too long"):
            svc.validate_chunk_content("x" * 2000)

    def test_validate_chunk_content_valid(self) -> None:
        svc = _make_service(AsyncMock())
        valid = "This is a valid chunk content with enough text. " * 5
        svc.validate_chunk_content(valid)

    def test_validate_chunk_content_manual_relaxed(self) -> None:
        svc = _make_service(AsyncMock())
        short_manual = "a" * 27
        svc.validate_chunk_content(short_manual, is_manual=True)
        with pytest.raises(ValidationError, match="too short"):
            svc.validate_chunk_content("ab", is_manual=True)


class TestComputeOwnerAndGroup:
    def test_internal_public(self) -> None:
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PUBLIC, None, 1)
        assert owner is None
        assert group is None

    def test_internal_group(self) -> None:
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_GROUP, 42, 1)
        assert owner is None
        assert group == 42

    def test_internal_private(self) -> None:
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PRIVATE, None, 1)
        assert owner == 1
        assert group is None


class TestDocumentCanEditChunks:
    def test_admin_can_edit(self) -> None:
        doc = Document(id=1, visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=100)
        assert doc.can_edit_chunks(1, UserRole.ADMIN) is True

    def test_owner_can_edit(self) -> None:
        doc = Document(id=1, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=100)
        assert doc.can_edit_chunks(100, UserRole.USER) is True

    def test_non_owner_cannot_edit(self) -> None:
        doc = Document(id=1, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=100)
        assert doc.can_edit_chunks(200, UserRole.USER) is False

    def test_group_doc_only_admin(self) -> None:
        doc = Document(id=1, visibility=DocumentVisibility.INTERNAL_GROUP, group_id=42, owner_id=None)
        assert doc.can_edit_chunks(1, UserRole.ADMIN) is True
        assert doc.can_edit_chunks(100, UserRole.USER) is False


# ---------------------------------------------------------------------------
# Characterization tests — edit_chunk
# ---------------------------------------------------------------------------


class TestEditChunk:
    @pytest.mark.asyncio
    async def test_edit_chunk_success(self) -> None:
        """edit_chunk enqueues both retrieval vectors through outbox and updates stats."""
        uow = _make_uow()
        svc = _make_service(uow)

        result = await svc.edit_chunk(
            document_id=1,
            chunk_id=10,
            content="new content " * 20,
            user_id=100,
            user_role=UserRole.ADMIN,
        )

        assert result.id == 10
        assert result.content == "new content " * 20
        assert result.edited_by == 100
        uow.chunks.update_content.assert_awaited_once()
        new_hash = content_hash(result.content)
        assert uow.chunks.update_content.call_args.kwargs["content_hash"] == new_hash
        uow.vector_outbox.enqueue.assert_awaited_once()
        enqueue_call = uow.vector_outbox.enqueue.call_args[0][0]
        assert enqueue_call.operation.value == "upsert_chunks"
        assert enqueue_call.payload["points"][0]["metadata"]["content_hash"] == new_hash
        uow.documents.set_has_manual_edits.assert_awaited_once_with(1, value=True)
        uow.documents.update_chunk_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_chunk_no_permission(self) -> None:
        """Non-owner cannot edit a private document."""
        doc = _make_doc(owner_id=1)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolationError, match="No permission"):
            await svc.edit_chunk(
                document_id=1,
                chunk_id=10,
                content="new " * 50,
                user_id=999,
                user_role=UserRole.USER,
            )

    @pytest.mark.asyncio
    async def test_edit_chunk_not_found(self) -> None:
        """Chunk not in DB raises EntityNotFoundError."""
        uow = _make_uow()
        uow.chunks.get_by_id = AsyncMock(return_value=None)
        svc = _make_service(uow)

        with pytest.raises(EntityNotFoundError):
            await svc.edit_chunk(
                document_id=1,
                chunk_id=999,
                content="new " * 50,
                user_id=100,
                user_role=UserRole.ADMIN,
            )

    @pytest.mark.asyncio
    async def test_edit_chunk_wrong_document(self) -> None:
        """Chunk belonging to a different document raises BusinessRuleViolationError."""
        chunk = _make_chunk(document_id=2)
        uow = _make_uow(chunk=chunk)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolationError, match="does not belong"):
            await svc.edit_chunk(
                document_id=1,
                chunk_id=10,
                content="new " * 50,
                user_id=100,
                user_role=UserRole.ADMIN,
            )

    @pytest.mark.asyncio
    async def test_edit_chunk_duplicate_warning(self) -> None:
        """Duplicate content hash produces a warning in the result."""
        uow = _make_uow()
        dup_chunk = _make_chunk(chunk_id=5)
        uow.chunks.find_duplicate_by_hash = AsyncMock(return_value=dup_chunk)
        svc = _make_service(uow)

        content = "x" * 300
        result = await svc.edit_chunk(
            document_id=1,
            chunk_id=10,
            content=content,
            user_id=100,
            user_role=UserRole.ADMIN,
        )

        assert result.warning is not None
        assert "existing chunk #5" in result.warning

    @pytest.mark.asyncio
    async def test_edit_without_old_hash_still_enqueues_vectors(self) -> None:
        """A missing old hash does not prevent replacing both vectors."""
        chunk = _make_chunk(content_hash=None)
        uow = _make_uow(chunk=chunk)
        svc = _make_service(uow)

        await svc.edit_chunk(
            document_id=1,
            chunk_id=10,
            content="new " * 50,
            user_id=100,
            user_role=UserRole.ADMIN,
        )
        uow.vector_outbox.enqueue.assert_awaited_once()


# ---------------------------------------------------------------------------
# Characterization tests — add_chunk
# ---------------------------------------------------------------------------


class TestAddChunk:
    @pytest.mark.asyncio
    async def test_add_chunk_success(self) -> None:
        """add_chunk inserts, enqueues outbox, calls BM25 add, updates stats."""
        uow = _make_uow(max_index=3)
        svc = _make_service(uow)

        result = await svc.add_chunk(
            document_id=1,
            content="new chunk text " * 20,
            user_id=100,
            user_role=UserRole.ADMIN,
        )

        assert result.document_id == 1
        assert result.chunk_index == 4
        assert result.content == "new chunk text " * 20
        assert result.manual is True
        uow.documents.get_by_id.assert_awaited_once_with(1, for_update=True)
        uow.chunks.insert_one.assert_awaited_once()
        uow.vector_outbox.enqueue.assert_awaited_once()
        enqueue_call = uow.vector_outbox.enqueue.call_args[0][0]
        assert enqueue_call.operation.value == "upsert_chunks"
        uow.documents.set_has_manual_edits.assert_awaited_once_with(1, value=True)
        uow.documents.update_chunk_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_chunk_wrong_status(self) -> None:
        """Cannot add chunks to a FAILED document."""
        doc = _make_doc(status=DocumentStatus.FAILED)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolationError, match="status"):
            await svc.add_chunk(
                document_id=1,
                content="new " * 50,
                user_id=100,
                user_role=UserRole.ADMIN,
            )

    @pytest.mark.asyncio
    async def test_add_chunk_no_permission(self) -> None:
        """Non-owner cannot add chunks to a private document."""
        doc = _make_doc(owner_id=1)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolationError, match="No permission"):
            await svc.add_chunk(
                document_id=1,
                content="new " * 50,
                user_id=999,
                user_role=UserRole.USER,
            )

    @pytest.mark.asyncio
    async def test_add_chunk_status_checked_before_permission(self) -> None:
        """Original behavior: status check runs before permission check.

        A user without permission on a FAILED document gets the status
        error, not the permission error.
        """
        doc = _make_doc(owner_id=1, status=DocumentStatus.FAILED)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolationError, match="status"):
            await svc.add_chunk(
                document_id=1,
                content="new " * 50,
                user_id=999,
                user_role=UserRole.USER,
            )

    @pytest.mark.asyncio
    async def test_add_chunk_duplicate_warning(self) -> None:
        """Duplicate content hash produces a warning."""
        dup = _make_chunk(chunk_id=7)
        uow = _make_uow()
        uow.chunks.find_duplicate_by_hash = AsyncMock(return_value=dup)
        svc = _make_service(uow)

        result = await svc.add_chunk(
            document_id=1,
            content="x" * 300,
            user_id=100,
            user_role=UserRole.ADMIN,
        )
        assert result.warning is not None
        assert "existing chunk #7" in result.warning

    @pytest.mark.asyncio
    async def test_add_chunk_with_page_and_section(self) -> None:
        """Page and section metadata are passed via outbox payload."""
        uow = _make_uow(max_index=0)
        svc = _make_service(uow)

        await svc.add_chunk(
            document_id=1,
            content="x" * 300,
            user_id=100,
            user_role=UserRole.ADMIN,
            page=3,
            section="Legal",
        )

        enqueue_call = uow.vector_outbox.enqueue.call_args[0][0]
        point = enqueue_call.payload["points"][0]
        assert point["metadata"]["page"] == 3
        assert point["metadata"]["section"] == "Legal"


# ---------------------------------------------------------------------------
# Characterization tests — delete_chunk
# ---------------------------------------------------------------------------


class TestDeleteChunk:
    @pytest.mark.asyncio
    async def test_delete_chunk_success(self) -> None:
        """delete_chunk removes SQL row and enqueues deletion of both vectors."""
        chunk = _make_chunk(content_hash="old_hash_abc")
        uow = _make_uow(chunk=chunk)
        svc = _make_service(uow)

        await svc.delete_chunk(document_id=1, chunk_id=10, user_id=100, user_role=UserRole.ADMIN)

        uow.chunks.delete_one.assert_awaited_once_with(10)
        enqueue_call = uow.vector_outbox.enqueue.call_args[0][0]
        assert enqueue_call.operation.value == "delete_chunks"
        uow.documents.update_chunk_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_chunk_no_permission(self) -> None:
        doc = _make_doc(owner_id=1)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolationError, match="No permission"):
            await svc.delete_chunk(document_id=1, chunk_id=10, user_id=999, user_role=UserRole.USER)

    @pytest.mark.asyncio
    async def test_delete_chunk_not_found(self) -> None:
        uow = _make_uow()
        uow.chunks.get_by_id = AsyncMock(return_value=None)
        svc = _make_service(uow)

        with pytest.raises(EntityNotFoundError):
            await svc.delete_chunk(document_id=1, chunk_id=999, user_id=100, user_role=UserRole.ADMIN)

    @pytest.mark.asyncio
    async def test_delete_chunk_wrong_document(self) -> None:
        chunk = _make_chunk(document_id=2)
        uow = _make_uow(chunk=chunk)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolationError, match="does not belong"):
            await svc.delete_chunk(document_id=1, chunk_id=10, user_id=100, user_role=UserRole.ADMIN)

    @pytest.mark.asyncio
    async def test_delete_without_hash_still_enqueues_vectors(self) -> None:
        """Point identity supports deletion without a content hash."""
        chunk = _make_chunk(content_hash=None)
        uow = _make_uow(chunk=chunk)
        svc = _make_service(uow)

        await svc.delete_chunk(document_id=1, chunk_id=10, user_id=100, user_role=UserRole.ADMIN)
        uow.vector_outbox.enqueue.assert_awaited_once()


# ---------------------------------------------------------------------------
# Characterization tests — create_manual_document
# ---------------------------------------------------------------------------


class TestCreateManualDocument:
    @pytest.mark.asyncio
    async def test_create_manual_document_success(self) -> None:
        """create_manual_document creates a Document with MANUAL source_type."""
        uow = _make_uow()
        saved_doc = _make_doc(id=42, source_type=SourceType.MANUAL.value, filename="My Manual")
        uow.documents.save = AsyncMock(return_value=saved_doc)
        svc = make_manual_service(uow)

        result = await svc.create_manual_document(
            title="My Manual",
            visibility=DocumentVisibility.INTERNAL_PUBLIC,
            user_id=100,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
        )

        assert result.id == 42
        assert result.source_type == SourceType.MANUAL.value
        uow.documents.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_manual_document_user_cannot_publish_public(self) -> None:
        """Only admin can create internal_public documents."""
        uow = _make_uow()
        svc = make_manual_service(uow)

        with pytest.raises(BusinessRuleViolationError):
            await svc.create_manual_document(
                title="Bad",
                visibility=DocumentVisibility.INTERNAL_PUBLIC,
                user_id=100,
                user_kind=UserKind.INTERNAL,
                user_role=UserRole.USER,
            )

    @pytest.mark.asyncio
    async def test_create_manual_document_group_requires_group_id(self) -> None:
        """Creating a group-visible document without group_id should fail."""
        uow = _make_uow()
        svc = make_manual_service(uow)

        with pytest.raises((ValidationError, BusinessRuleViolationError)):
            await svc.create_manual_document(
                title="Bad",
                visibility=DocumentVisibility.INTERNAL_GROUP,
                user_id=100,
                user_kind=UserKind.INTERNAL,
                user_role=UserRole.USER,
                group_id=None,
            )
