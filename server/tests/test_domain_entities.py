"""Tests for domain entities: User, Document, Conversation, Message, Chunk.

Pure unit tests — no infrastructure dependencies. All business logic rules are tested
following AAA (Arrange-Act-Assert) pattern.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app"))

from domain.entities.chunk import Chunk
from domain.entities.conversation import Conversation
from domain.entities.document import Document
from domain.entities.message import Message
from domain.entities.user import User
from domain.exceptions import BusinessRuleViolation
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility


# ---------------------------------------------------------------------------
# Helpers / Factories
# ---------------------------------------------------------------------------

def _make_user(**overrides) -> User:
    defaults = dict(id=1, email="test@example.com", role=UserRole.USER, kind=UserKind.INTERNAL, is_active=True)
    defaults.update(overrides)
    return User(**defaults)


def _make_document(**overrides) -> Document:
    defaults = dict(id=1, filename="doc.pdf", source_path="/tmp/doc.pdf", visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=1)
    defaults.update(overrides)
    return Document(**defaults)


def _make_conversation(**overrides) -> Conversation:
    defaults = dict(id=1, user_id=10)
    defaults.update(overrides)
    return Conversation(**defaults)


def _make_message(**overrides) -> Message:
    defaults = dict(id=1, conversation_id=1, role=MessageRole.USER, content="hello")
    defaults.update(overrides)
    return Message(**defaults)


# ===========================================================================
# User Entity Tests
# ===========================================================================


class TestUserCreation:
    """Business rule: only admin can create users."""

    def test_admin_can_create_users(self):
        # Arrange
        user = _make_user()
        # Act & Assert — should not raise
        user.can_be_created_by(UserRole.ADMIN)

    def test_non_admin_cannot_create_users(self):
        # Arrange
        user = _make_user()
        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="Only admin can create users"):
            user.can_be_created_by(UserRole.USER)

    def test_admin_role_can_create_users(self):
        # Arrange & Act & Assert
        User().can_be_created_by(UserRole.ADMIN)

    def test_user_role_cannot_create_users(self):
        # Arrange & Act & Assert
        with pytest.raises(BusinessRuleViolation):
            User().can_be_created_by(UserRole.USER)


class TestUserEnsureValidForCreation:
    """Business rule: client cannot be admin."""

    def test_client_with_admin_role_raises(self):
        # Arrange
        user = _make_user(kind=UserKind.CLIENT, role=UserRole.ADMIN)
        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="Client cannot be admin"):
            user.ensure_valid_for_creation()

    def test_client_with_user_role_is_valid(self):
        # Arrange
        user = _make_user(kind=UserKind.CLIENT, role=UserRole.USER)
        # Act & Assert — should not raise
        user.ensure_valid_for_creation()

    def test_internal_with_admin_role_is_valid(self):
        # Arrange
        user = _make_user(kind=UserKind.INTERNAL, role=UserRole.ADMIN)
        # Act & Assert
        user.ensure_valid_for_creation()

    def test_internal_with_user_role_is_valid(self):
        # Arrange
        user = _make_user(kind=UserKind.INTERNAL, role=UserRole.USER)
        # Act & Assert
        user.ensure_valid_for_creation()


class TestUserDeactivateSelfProhibited:
    """Business rule: admin cannot deactivate themselves."""

    def test_self_deactivation_raises(self):
        # Arrange
        user = _make_user(id=42)
        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="Cannot deactivate yourself"):
            user.deactivate_self_prohibited(42)

    def test_deactivating_other_user_is_allowed(self):
        # Arrange
        user = _make_user(id=42)
        # Act & Assert — should not raise
        user.deactivate_self_prohibited(99)

    def test_deactivating_other_user_with_same_role(self):
        # Arrange
        user = _make_user(id=10, role=UserRole.USER)
        # Act & Assert
        user.deactivate_self_prohibited(20)

    def test_self_deactivation_with_different_id(self):
        # Arrange
        user = _make_user(id=100)
        # Act & Assert
        user.deactivate_self_prohibited(200)


class TestUserToggleActive:
    def test_toggle_active_true(self):
        # Arrange
        user = _make_user(is_active=False)
        # Act
        user.toggle_active(True)
        # Assert
        assert user.is_active is True

    def test_toggle_active_false(self):
        # Arrange
        user = _make_user(is_active=True)
        # Act
        user.toggle_active(False)
        # Assert
        assert user.is_active is False

    def test_toggle_active_idempotent(self):
        # Arrange
        user = _make_user(is_active=True)
        # Act
        user.toggle_active(True)
        user.toggle_active(True)
        # Assert
        assert user.is_active is True


class TestUserChangeRole:
    def test_change_role_user_to_admin(self):
        # Arrange
        user = _make_user(role=UserRole.USER, kind=UserKind.INTERNAL)
        # Act
        user.change_role(UserRole.ADMIN)
        # Assert
        assert user.role == UserRole.ADMIN

    def test_change_role_admin_to_user(self):
        # Arrange
        user = _make_user(role=UserRole.ADMIN)
        # Act
        user.change_role(UserRole.USER)
        # Assert
        assert user.role == UserRole.USER

    def test_client_cannot_become_admin(self):
        # Arrange
        user = _make_user(kind=UserKind.CLIENT, role=UserRole.USER)
        # Act & Assert
        with pytest.raises(BusinessRuleViolation, match="Client cannot be admin"):
            user.change_role(UserRole.ADMIN)

    def test_client_can_stay_user(self):
        # Arrange
        user = _make_user(kind=UserKind.CLIENT, role=UserRole.USER)
        # Act & Assert — should not raise
        user.change_role(UserRole.USER)


class TestUserPostInit:
    def test_role_from_string(self):
        # Arrange & Act
        user = User(role="admin")
        # Assert
        assert user.role == UserRole.ADMIN

    def test_kind_from_string(self):
        # Arrange & Act
        user = User(kind="client")
        # Assert
        assert user.kind == UserKind.CLIENT

    def test_invalid_role_string_raises(self):
        # Arrange & Act & Assert
        with pytest.raises(Exception):
            User(role="superadmin")

    def test_invalid_kind_string_raises(self):
        # Arrange & Act & Assert
        with pytest.raises(Exception):
            User(kind="external")


class TestUserDefaults:
    def test_default_values(self):
        # Arrange & Act
        user = User()
        # Assert
        assert user.id is None
        assert user.email == ""
        assert user.hashed_password == ""
        assert user.role == UserRole.USER
        assert user.kind == UserKind.INTERNAL
        assert user.is_active is True
        assert user.group_ids == []
        assert user.assigned_client_ids == []


# ===========================================================================
# Document Entity Tests
# ===========================================================================


class TestDocumentMarkProcessing:
    def test_mark_processing(self):
        # Arrange
        doc = _make_document(status=DocumentStatus.PENDING)
        # Act
        doc.mark_processing()
        # Assert
        assert doc.status == DocumentStatus.PROCESSING

    def test_mark_processing_from_done(self):
        # Arrange
        doc = _make_document(status=DocumentStatus.DONE)
        # Act
        doc.mark_processing()
        # Assert
        assert doc.status == DocumentStatus.PROCESSING


class TestDocumentMarkDone:
    def test_mark_done_sets_status_chunks_chars(self):
        # Arrange
        doc = _make_document(status=DocumentStatus.PROCESSING)
        # Act
        doc.mark_done(chunks=10, chars=5000)
        # Assert
        assert doc.status == DocumentStatus.DONE
        assert doc.chunks == 10
        assert doc.chars == 5000
        assert doc.indexed_at is not None

    def test_mark_done_updates_indexed_at(self):
        # Arrange
        doc = _make_document(indexed_at=None)
        before = datetime.now(UTC)
        # Act
        doc.mark_done(chunks=5, chars=1000)
        # Assert
        assert doc.indexed_at >= before


class TestDocumentMarkFailed:
    def test_mark_failed_sets_error(self):
        # Arrange
        doc = _make_document(status=DocumentStatus.PROCESSING)
        # Act
        doc.mark_failed("OCR failed")
        # Assert
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message == "OCR failed"

    def test_mark_failed_clears_previous_error(self):
        # Arrange
        doc = _make_document(error_message="old error")
        # Act
        doc.mark_failed("new error")
        # Assert
        assert doc.error_message == "new error"


class TestDocumentCanBeDeletedBy:
    def test_owner_can_delete(self):
        # Arrange
        doc = _make_document(owner_id=10)
        # Act & Assert
        assert doc.can_be_deleted_by(10, UserRole.USER) is True

    def test_admin_can_delete_any(self):
        # Arrange
        doc = _make_document(owner_id=10)
        # Act & Assert
        assert doc.can_be_deleted_by(99, UserRole.ADMIN) is True

    def test_non_owner_non_admin_cannot_delete(self):
        # Arrange
        doc = _make_document(owner_id=10)
        # Act & Assert
        assert doc.can_be_deleted_by(20, UserRole.USER) is False

    def test_owner_admin_can_also_delete(self):
        # Arrange
        doc = _make_document(owner_id=10)
        # Act & Assert
        assert doc.can_be_deleted_by(10, UserRole.ADMIN) is True


class TestDocumentPostInit:
    def test_visibility_from_string(self):
        # Arrange & Act
        doc = Document(visibility="internal_private")
        # Assert
        assert doc.visibility == DocumentVisibility.INTERNAL_PRIVATE

    def test_status_from_string(self):
        # Arrange & Act
        doc = Document(status="done")
        # Assert
        assert doc.status == DocumentStatus.DONE

    def test_invalid_visibility_string_raises(self):
        # Arrange & Act & Assert
        with pytest.raises(Exception):
            Document(visibility="public")


class TestDocumentDefaults:
    def test_default_values(self):
        # Arrange & Act
        doc = Document()
        # Assert
        assert doc.id is None
        assert doc.filename == ""
        assert doc.source_path == ""
        assert doc.visibility == DocumentVisibility.INTERNAL_PUBLIC
        assert doc.owner_id is None
        assert doc.group_id is None
        assert doc.status == DocumentStatus.PENDING
        assert doc.error_message is None
        assert doc.chunks is None
        assert doc.chars is None
        assert doc.indexed_at is None


# ===========================================================================
# Conversation Entity Tests
# ===========================================================================


class TestConversationIsOwnedBy:
    def test_owner_matches(self):
        # Arrange
        conv = _make_conversation(user_id=10)
        # Act & Assert
        assert conv.is_owned_by(10) is True

    def test_owner_does_not_match(self):
        # Arrange
        conv = _make_conversation(user_id=10)
        # Act & Assert
        assert conv.is_owned_by(20) is False

    def test_ownership_with_zero_user_id(self):
        # Arrange
        conv = _make_conversation(user_id=0)
        # Act & Assert
        assert conv.is_owned_by(0) is True
        assert conv.is_owned_by(1) is False


class TestConversationAddMessage:
    def test_add_single_message(self):
        # Arrange
        conv = _make_conversation()
        msg = _make_message()
        # Act
        conv.add_message(msg)
        # Assert
        assert len(conv.messages) == 1
        assert conv.messages[0] is msg

    def test_add_multiple_messages(self):
        # Arrange
        conv = _make_conversation()
        msg1 = _make_message(id=1, content="first")
        msg2 = _make_message(id=2, content="second")
        # Act
        conv.add_message(msg1)
        conv.add_message(msg2)
        # Assert
        assert len(conv.messages) == 2

    def test_add_message_preserves_order(self):
        # Arrange
        conv = _make_conversation()
        msgs = [_make_message(id=i, content=f"msg{i}") for i in range(5)]
        # Act
        for m in msgs:
            conv.add_message(m)
        # Assert
        assert [m.content for m in conv.messages] == ["msg0", "msg1", "msg2", "msg3", "msg4"]


class TestConversationDefaults:
    def test_default_values(self):
        # Arrange & Act
        conv = Conversation()
        # Assert
        assert conv.id is None
        assert conv.user_id == 0
        assert conv.messages == []


# ===========================================================================
# Message Entity Tests
# ===========================================================================


class TestMessagePostInit:
    def test_role_from_string_user(self):
        # Arrange & Act
        msg = Message(role="user")
        # Assert
        assert msg.role == MessageRole.USER

    def test_role_from_string_assistant(self):
        # Arrange & Act
        msg = Message(role="assistant")
        # Assert
        assert msg.role == MessageRole.ASSISTANT

    def test_invalid_role_string_raises(self):
        # Arrange & Act & Assert
        with pytest.raises(Exception):
            Message(role="system")


class TestMessageDefaults:
    def test_default_values(self):
        # Arrange & Act
        msg = Message()
        # Assert
        assert msg.id is None
        assert msg.conversation_id == 0
        assert msg.role == MessageRole.USER
        assert msg.content == ""
        assert msg.sources == []
        assert msg.created_at is not None


class TestMessageSources:
    def test_message_with_sources(self):
        # Arrange
        sources = [{"source": "doc.pdf", "page": 1}]
        # Act
        msg = Message(sources=sources)
        # Assert
        assert msg.sources == sources

    def test_message_empty_sources(self):
        # Arrange & Act
        msg = Message(sources=[])
        # Assert
        assert msg.sources == []


# ===========================================================================
# Chunk Entity Tests
# ===========================================================================


class TestChunk:
    def test_chunk_defaults(self):
        # Arrange & Act
        chunk = Chunk()
        # Assert
        assert chunk.content == ""
        assert chunk.metadata == {}
        assert chunk.score is None

    def test_chunk_with_values(self):
        # Arrange & Act
        chunk = Chunk(content="text", metadata={"page": 1}, score=0.95)
        # Assert
        assert chunk.content == "text"
        assert chunk.metadata == {"page": 1}
        assert chunk.score == 0.95

    def test_chunk_metadata_mutable(self):
        # Arrange
        chunk = Chunk()
        # Act
        chunk.metadata["key"] = "value"
        # Assert
        assert chunk.metadata == {"key": "value"}
