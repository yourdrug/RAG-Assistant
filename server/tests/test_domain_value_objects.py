"""Tests for domain value objects: UserRole, UserKind, DocumentVisibility,
DocumentStatus, MessageRole.

Pure unit tests — no infrastructure dependencies. All validation logic is tested.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app"))

from domain.exceptions import ValidationError
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility

# ===========================================================================
# UserRole Tests
# ===========================================================================


class TestUserRole:
    def test_valid_admin(self):
        assert UserRole("admin") == UserRole.ADMIN

    def test_valid_user(self):
        assert UserRole("user") == UserRole.USER

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            UserRole("superadmin")

    def test_validate_valid_admin(self):
        assert UserRole.validate("admin") == UserRole.ADMIN

    def test_validate_valid_user(self):
        assert UserRole.validate("user") == UserRole.USER

    def test_validate_invalid_role_raises_validation_error(self):
        with pytest.raises(ValidationError, match="role must be 'admin' or 'user'"):
            UserRole.validate("moderator")

    def test_validate_empty_string_raises(self):
        with pytest.raises(ValidationError):
            UserRole.validate("")

    def test_role_values(self):
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.USER.value == "user"

    def test_role_is_str_enum(self):
        assert isinstance(UserRole.ADMIN, str)
        assert isinstance(UserRole.USER, str)


# ===========================================================================
# UserKind Tests
# ===========================================================================


class TestUserKind:
    def test_valid_internal(self):
        assert UserKind("internal") == UserKind.INTERNAL

    def test_valid_client(self):
        assert UserKind("client") == UserKind.CLIENT

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            UserKind("external")

    def test_validate_valid_internal(self):
        assert UserKind.validate("internal") == UserKind.INTERNAL

    def test_validate_valid_client(self):
        assert UserKind.validate("client") == UserKind.CLIENT

    def test_validate_invalid_kind_raises_validation_error(self):
        with pytest.raises(ValidationError, match="kind must be 'internal' or 'client'"):
            UserKind.validate("partner")

    def test_validate_empty_string_raises(self):
        with pytest.raises(ValidationError):
            UserKind.validate("")

    def test_kind_values(self):
        assert UserKind.INTERNAL.value == "internal"
        assert UserKind.CLIENT.value == "client"

    def test_kind_is_str_enum(self):
        assert isinstance(UserKind.INTERNAL, str)
        assert isinstance(UserKind.CLIENT, str)


# ===========================================================================
# DocumentVisibility Tests
# ===========================================================================


class TestDocumentVisibility:
    def test_all_values_exist(self):
        assert DocumentVisibility.INTERNAL_PUBLIC.value == "internal_public"
        assert DocumentVisibility.INTERNAL_GROUP.value == "internal_group"
        assert DocumentVisibility.INTERNAL_PRIVATE.value == "internal_private"
        assert DocumentVisibility.CLIENT_PRIVATE.value == "client_private"

    def test_validate_valid_values(self):
        for v in ["internal_public", "internal_group", "internal_private", "client_private"]:
            assert DocumentVisibility.validate(v) == DocumentVisibility(v)

    def test_validate_invalid_value_raises(self):
        with pytest.raises(ValidationError, match="visibility must be one of"):
            DocumentVisibility.validate("public")

    def test_validate_empty_string_raises(self):
        with pytest.raises(ValidationError):
            DocumentVisibility.validate("")

    def test_from_string(self):
        vis = DocumentVisibility("internal_group")
        assert vis == DocumentVisibility.INTERNAL_GROUP


# ===========================================================================
# DocumentStatus Tests
# ===========================================================================


class TestDocumentStatus:
    def test_all_values(self):
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.DONE.value == "done"
        assert DocumentStatus.FAILED.value == "failed"

    def test_from_string(self):
        assert DocumentStatus("pending") == DocumentStatus.PENDING
        assert DocumentStatus("done") == DocumentStatus.DONE

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            DocumentStatus("unknown")


# ===========================================================================
# MessageRole Tests
# ===========================================================================


class TestMessageRole:
    def test_valid_user(self):
        assert MessageRole("user") == MessageRole.USER

    def test_valid_assistant(self):
        assert MessageRole("assistant") == MessageRole.ASSISTANT

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            MessageRole("system")

    def test_values(self):
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


# ===========================================================================
# Cross-VO consistency tests
# ===========================================================================


class TestValueObjectConsistency:
    """Ensure value objects behave consistently as StrEnums."""

    def test_role_string_comparison(self):
        assert UserRole.ADMIN == "admin"
        assert UserRole.USER == "user"

    def test_kind_string_comparison(self):
        assert UserKind.INTERNAL == "internal"
        assert UserKind.CLIENT == "client"

    def test_visibility_string_comparison(self):
        assert DocumentVisibility.INTERNAL_PUBLIC == "internal_public"

    def test_status_string_comparison(self):
        assert DocumentStatus.PENDING == "pending"

    def test_role_in_set(self):
        roles = {UserRole.ADMIN, UserRole.USER}
        assert "admin" in roles
        assert "user" in roles

    def test_kind_in_set(self):
        kinds = {UserKind.INTERNAL, UserKind.CLIENT}
        assert "internal" in kinds
        assert "client" in kinds
