"""Tests for ChunkService business logic."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from application.services.chunk_service import ChunkService
from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.repositories.chunk_repository import ChunkSearchResult, ChunkStats
from domain.services import compute_owner_and_group
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserRole
from domain.value_objects.source_type import SourceType
from domain.value_objects.visibility import DocumentVisibility


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeChunkSettings:
    chunk_size: int = 550


def _make_doc(**overrides) -> Document:
    defaults = {
        "id": 1,
        "filename": "test.pdf",
        "visibility": DocumentVisibility.INTERNAL_PUBLIC,
        "owner_id": None,
        "status": DocumentStatus.DONE,
        "source_type": SourceType.FILE.value,
    }
    defaults.update(overrides)
    return Document(**defaults)


def _make_chunk(**overrides) -> ChunkSearchResult:
    defaults = {
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
    return ChunkSearchResult(**defaults)


def _make_uow(doc=None, chunk=None, max_index=0):
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
    return uow


def _make_service(uow, *, chunk_settings=None, bm25_index=None):
    factory = MagicMock()
    factory.create.return_value = AsyncMock(
        __aenter__=AsyncMock(return_value=uow),
        __aexit__=AsyncMock(return_value=False),
    )
    return ChunkService(
        uow_factory=factory,
        vector_store_repo=AsyncMock(),
        chunk_settings=chunk_settings or _FakeChunkSettings(),
        bm25_index=bm25_index or MagicMock(),
    )


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


class TestChunkServiceValidation:
    """Tests for ChunkMutationService validation helpers (via ChunkService._mutation)."""

    def test_validate_chunk_content_empty(self):
        svc = _make_service(AsyncMock())
        with pytest.raises(ValidationError, match="cannot be empty"):
            svc._mutation._validate_chunk_content("")
        with pytest.raises(ValidationError, match="cannot be empty"):
            svc._mutation._validate_chunk_content("   ")

    def test_validate_chunk_content_too_short(self):
        svc = _make_service(AsyncMock())
        with pytest.raises(ValidationError, match="too short"):
            svc._mutation._validate_chunk_content("ab")

    def test_validate_chunk_content_too_long(self):
        svc = _make_service(AsyncMock())
        with pytest.raises(ValidationError, match="too long"):
            svc._mutation._validate_chunk_content("x" * 2000)

    def test_validate_chunk_content_valid(self):
        svc = _make_service(AsyncMock())
        valid = "This is a valid chunk content with enough text. " * 5
        svc._mutation._validate_chunk_content(valid)

    def test_validate_chunk_content_manual_relaxed(self):
        svc = _make_service(AsyncMock())
        short_manual = "a" * 27
        svc._mutation._validate_chunk_content(short_manual, is_manual=True)
        with pytest.raises(ValidationError, match="too short"):
            svc._mutation._validate_chunk_content("ab", is_manual=True)


class TestComputeOwnerAndGroup:
    def test_internal_public(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PUBLIC, None, 1)
        assert owner is None and group is None

    def test_internal_group(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_GROUP, 42, 1)
        assert owner is None and group == 42

    def test_internal_private(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PRIVATE, None, 1)
        assert owner == 1 and group is None


class TestDocumentCanEditChunks:
    def test_admin_can_edit(self):
        doc = Document(id=1, visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=100)
        assert doc.can_edit_chunks(1, UserRole.ADMIN) is True

    def test_owner_can_edit(self):
        doc = Document(id=1, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=100)
        assert doc.can_edit_chunks(100, UserRole.USER) is True

    def test_non_owner_cannot_edit(self):
        doc = Document(id=1, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=100)
        assert doc.can_edit_chunks(200, UserRole.USER) is False

    def test_group_doc_only_admin(self):
        doc = Document(id=1, visibility=DocumentVisibility.INTERNAL_GROUP, group_id=42, owner_id=None)
        assert doc.can_edit_chunks(1, UserRole.ADMIN) is True
        assert doc.can_edit_chunks(100, UserRole.USER) is False


# ---------------------------------------------------------------------------
# Characterization tests — edit_chunk
# ---------------------------------------------------------------------------


class TestEditChunk:
    @pytest.mark.asyncio
    async def test_edit_chunk_success(self):
        """edit_chunk enqueues outbox UPSERT, calls BM25 replace, updates stats."""
        uow = _make_uow()
        bm25 = MagicMock()
        svc = _make_service(uow, bm25_index=bm25)

        result = await svc.edit_chunk(
            document_id=1,
            chunk_id=10,
            content="new content " * 20,
            user_id=100,
            user_role="admin",
        )

        assert result.id == 10
        assert result.content == "new content " * 20
        assert result.edited_by == 100
        uow.chunks.update_content.assert_awaited_once()
        uow.vector_outbox.enqueue.assert_awaited_once()
        enqueue_call = uow.vector_outbox.enqueue.call_args[0][0]
        assert enqueue_call.operation.value == "upsert_chunks"
        bm25.replace.assert_called_once()
        uow.documents.set_has_manual_edits.assert_awaited_once_with(1, True)
        uow.documents.update_chunk_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_chunk_no_permission(self):
        """Non-owner cannot edit a private document."""
        doc = _make_doc(owner_id=1)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolation, match="No permission"):
            await svc.edit_chunk(
                document_id=1,
                chunk_id=10,
                content="new " * 50,
                user_id=999,
                user_role="user",
            )

    @pytest.mark.asyncio
    async def test_edit_chunk_not_found(self):
        """Chunk not in DB raises EntityNotFound."""
        uow = _make_uow()
        uow.chunks.get_by_id = AsyncMock(return_value=None)
        svc = _make_service(uow)

        with pytest.raises(EntityNotFound):
            await svc.edit_chunk(
                document_id=1,
                chunk_id=999,
                content="new " * 50,
                user_id=100,
                user_role="admin",
            )

    @pytest.mark.asyncio
    async def test_edit_chunk_wrong_document(self):
        """Chunk belonging to a different document raises BusinessRuleViolation."""
        chunk = _make_chunk(document_id=2)
        uow = _make_uow(chunk=chunk)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolation, match="does not belong"):
            await svc.edit_chunk(
                document_id=1,
                chunk_id=10,
                content="new " * 50,
                user_id=100,
                user_role="admin",
            )

    @pytest.mark.asyncio
    async def test_edit_chunk_duplicate_warning(self):
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
            user_role="admin",
        )

        assert result.warning is not None
        assert "existing chunk #5" in result.warning

    @pytest.mark.asyncio
    async def test_edit_chunk_no_bm25_when_no_old_hash(self):
        """When chunk has no old content_hash, BM25 replace is skipped."""
        chunk = _make_chunk(content_hash=None)
        uow = _make_uow(chunk=chunk)
        bm25 = MagicMock()
        svc = _make_service(uow, bm25_index=bm25)

        await svc.edit_chunk(
            document_id=1,
            chunk_id=10,
            content="new " * 50,
            user_id=100,
            user_role="admin",
        )
        bm25.replace.assert_not_called()


# ---------------------------------------------------------------------------
# Characterization tests — add_chunk
# ---------------------------------------------------------------------------


class TestAddChunk:
    @pytest.mark.asyncio
    async def test_add_chunk_success(self):
        """add_chunk inserts, enqueues outbox, calls BM25 add, updates stats."""
        uow = _make_uow(max_index=3)
        bm25 = MagicMock()
        svc = _make_service(uow, bm25_index=bm25)

        result = await svc.add_chunk(
            document_id=1,
            content="new chunk text " * 20,
            user_id=100,
            user_role="admin",
        )

        assert result.document_id == 1
        assert result.chunk_index == 4
        assert result.content == "new chunk text " * 20
        assert result.manual is True
        uow.chunks.insert_one.assert_awaited_once()
        uow.vector_outbox.enqueue.assert_awaited_once()
        enqueue_call = uow.vector_outbox.enqueue.call_args[0][0]
        assert enqueue_call.operation.value == "upsert_chunks"
        bm25.add.assert_called_once()
        uow.documents.set_has_manual_edits.assert_awaited_once_with(1, True)
        uow.documents.update_chunk_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_chunk_wrong_status(self):
        """Cannot add chunks to a FAILED document."""
        doc = _make_doc(status=DocumentStatus.FAILED)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolation, match="status"):
            await svc.add_chunk(
                document_id=1,
                content="new " * 50,
                user_id=100,
                user_role="admin",
            )

    @pytest.mark.asyncio
    async def test_add_chunk_no_permission(self):
        """Non-owner cannot add chunks to a private document."""
        doc = _make_doc(owner_id=1)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolation, match="No permission"):
            await svc.add_chunk(
                document_id=1,
                content="new " * 50,
                user_id=999,
                user_role="user",
            )

    @pytest.mark.asyncio
    async def test_add_chunk_status_checked_before_permission(self):
        """Original behavior: status check runs before permission check.

        A user without permission on a FAILED document gets the status
        error, not the permission error.
        """
        doc = _make_doc(owner_id=1, status=DocumentStatus.FAILED)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolation, match="status"):
            await svc.add_chunk(
                document_id=1,
                content="new " * 50,
                user_id=999,
                user_role="user",
            )

    @pytest.mark.asyncio
    async def test_add_chunk_duplicate_warning(self):
        """Duplicate content hash produces a warning."""
        dup = _make_chunk(chunk_id=7)
        uow = _make_uow()
        uow.chunks.find_duplicate_by_hash = AsyncMock(return_value=dup)
        svc = _make_service(uow)

        result = await svc.add_chunk(
            document_id=1,
            content="x" * 300,
            user_id=100,
            user_role="admin",
        )
        assert result.warning is not None
        assert "existing chunk #7" in result.warning

    @pytest.mark.asyncio
    async def test_add_chunk_with_page_and_section(self):
        """Page and section metadata are passed via outbox payload."""
        uow = _make_uow(max_index=0)
        svc = _make_service(uow)

        await svc.add_chunk(
            document_id=1,
            content="x" * 300,
            user_id=100,
            user_role="admin",
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
    async def test_delete_chunk_success(self):
        """delete_chunk removes DB row, enqueues outbox DELETE, calls BM25 remove."""
        chunk = _make_chunk(content_hash="old_hash_abc")
        uow = _make_uow(chunk=chunk)
        bm25 = MagicMock()
        svc = _make_service(uow, bm25_index=bm25)

        await svc.delete_chunk(document_id=1, chunk_id=10, user_id=100, user_role="admin")

        uow.chunks.delete_one.assert_awaited_once_with(10)
        enqueue_call = uow.vector_outbox.enqueue.call_args[0][0]
        assert enqueue_call.operation.value == "delete_chunks"
        bm25.remove.assert_called_once_with("old_hash_abc")
        uow.documents.update_chunk_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_chunk_no_permission(self):
        doc = _make_doc(owner_id=1)
        uow = _make_uow(doc=doc)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolation, match="No permission"):
            await svc.delete_chunk(document_id=1, chunk_id=10, user_id=999, user_role="user")

    @pytest.mark.asyncio
    async def test_delete_chunk_not_found(self):
        uow = _make_uow()
        uow.chunks.get_by_id = AsyncMock(return_value=None)
        svc = _make_service(uow)

        with pytest.raises(EntityNotFound):
            await svc.delete_chunk(document_id=1, chunk_id=999, user_id=100, user_role="admin")

    @pytest.mark.asyncio
    async def test_delete_chunk_wrong_document(self):
        chunk = _make_chunk(document_id=2)
        uow = _make_uow(chunk=chunk)
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolation, match="does not belong"):
            await svc.delete_chunk(document_id=1, chunk_id=10, user_id=100, user_role="admin")

    @pytest.mark.asyncio
    async def test_delete_chunk_no_bm25_when_no_hash(self):
        """When chunk has no content_hash, BM25 remove is skipped."""
        chunk = _make_chunk(content_hash=None)
        uow = _make_uow(chunk=chunk)
        bm25 = MagicMock()
        svc = _make_service(uow, bm25_index=bm25)

        await svc.delete_chunk(document_id=1, chunk_id=10, user_id=100, user_role="admin")
        bm25.remove.assert_not_called()


# ---------------------------------------------------------------------------
# Characterization tests — create_manual_document
# ---------------------------------------------------------------------------


class TestCreateManualDocument:
    @pytest.mark.asyncio
    async def test_create_manual_document_success(self):
        """create_manual_document creates a Document with MANUAL source_type."""
        uow = _make_uow()
        saved_doc = _make_doc(id=42, source_type=SourceType.MANUAL.value, filename="My Manual")
        uow.documents.save = AsyncMock(return_value=saved_doc)
        svc = _make_service(uow)

        result = await svc.create_manual_document(
            title="My Manual",
            visibility="internal_public",
            user_id=100,
            user_kind="internal",
            user_role="admin",
        )

        assert result.id == 42
        assert result.source_type == SourceType.MANUAL.value
        uow.documents.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_manual_document_user_cannot_publish_public(self):
        """Only admin can create internal_public documents."""
        uow = _make_uow()
        svc = _make_service(uow)

        with pytest.raises(BusinessRuleViolation):
            await svc.create_manual_document(
                title="Bad",
                visibility="internal_public",
                user_id=100,
                user_kind="internal",
                user_role="user",
            )

    @pytest.mark.asyncio
    async def test_create_manual_document_group_requires_group_id(self):
        """Creating a group-visible document without group_id should fail."""
        uow = _make_uow()
        svc = _make_service(uow)

        with pytest.raises((ValidationError, BusinessRuleViolation)):
            await svc.create_manual_document(
                title="Bad",
                visibility="internal_group",
                user_id=100,
                user_kind="internal",
                user_role="user",
                group_id=None,
            )
