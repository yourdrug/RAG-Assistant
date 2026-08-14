"""Tests for ChunkService business logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import UTC, datetime

from application.services.chunk_service import ChunkService
from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserRole
from domain.value_objects.visibility import DocumentVisibility


@pytest.fixture
def mock_uow_factory():
    """Mock UnitOfWorkFactory."""
    return AsyncMock()


@pytest.fixture
def mock_vector_store():
    """Mock VectorStoreRepository."""
    mock = AsyncMock()
    mock.generate_embeddings.return_value = [0.1] * 128
    return mock


@pytest.fixture
def chunk_service(mock_uow_factory, mock_vector_store):
    """Create ChunkService with mocked dependencies."""
    return ChunkService(
        uow_factory=mock_uow_factory,
        vector_store_repo=mock_vector_store,
    )


class TestChunkService:
    """Tests for ChunkService."""

    def test_validate_chunk_content_empty(self, chunk_service):
        """Test that empty content raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            chunk_service._validate_chunk_content("")

        with pytest.raises(ValidationError, match="cannot be empty"):
            chunk_service._validate_chunk_content("   ")

    def test_validate_chunk_content_too_short(self, chunk_service):
        """Test that very short content raises ValidationError."""
        with pytest.raises(ValidationError, match="too short"):
            chunk_service._validate_chunk_content("ab")

    def test_validate_chunk_content_too_long(self, chunk_service):
        """Test that very long content raises ValidationError."""
        with pytest.raises(ValidationError, match="too long"):
            chunk_service._validate_chunk_content("x" * 2000)

    def test_validate_chunk_content_valid(self, chunk_service):
        """Test that valid content passes validation."""
        # Content must be at least 0.3 * chunk_size (default 550) = 165 chars
        valid_content = "This is a valid chunk content with enough text. " * 5  # ~235 chars
        # Should not raise
        chunk_service._validate_chunk_content(valid_content)

    def test_compute_owner_and_group(self):
        """Test owner and group computation based on visibility."""
        # Internal public
        owner, group = ChunkService._compute_owner_and_group(
            DocumentVisibility.INTERNAL_PUBLIC, None, 1
        )
        assert owner is None
        assert group is None

        # Internal group
        owner, group = ChunkService._compute_owner_and_group(
            DocumentVisibility.INTERNAL_GROUP, 42, 1
        )
        assert owner is None
        assert group == 42

        # Internal private
        owner, group = ChunkService._compute_owner_and_group(
            DocumentVisibility.INTERNAL_PRIVATE, None, 1
        )
        assert owner == 1
        assert group is None


class TestDocumentCanEditChunks:
    """Tests for Document.can_edit_chunks method."""

    def test_admin_can_edit(self):
        """Test that admin can edit any document."""
        doc = Document(
            id=1,
            visibility=DocumentVisibility.INTERNAL_PUBLIC,
            owner_id=100,
        )
        assert doc.can_edit_chunks(1, UserRole.ADMIN) is True

    def test_owner_can_edit(self):
        """Test that owner can edit their own document."""
        doc = Document(
            id=1,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=100,
        )
        assert doc.can_edit_chunks(100, UserRole.USER) is True

    def test_non_owner_cannot_edit(self):
        """Test that non-owner cannot edit private document."""
        doc = Document(
            id=1,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=100,
        )
        assert doc.can_edit_chunks(200, UserRole.USER) is False

    def test_group_doc_only_admin(self):
        """Test that group documents can only be edited by admin."""
        doc = Document(
            id=1,
            visibility=DocumentVisibility.INTERNAL_GROUP,
            group_id=42,
            owner_id=None,
        )
        # Admin can edit
        assert doc.can_edit_chunks(1, UserRole.ADMIN) is True
        # Regular user cannot edit group docs
        assert doc.can_edit_chunks(100, UserRole.USER) is False
