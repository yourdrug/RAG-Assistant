"""Tests for domain/services/access_control.py — pure business rules for document visibility.

Tests three functions:
  - validate_document_visibility
  - compute_owner_and_group
  - can_view_document

All tests are pure unit tests with no infrastructure dependencies.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app"))

from domain.exceptions import BusinessRuleViolation, ValidationError
from domain.services.access_control import (
    ALLOWED_VISIBILITY_FOR_KIND,
    can_view_document,
    compute_owner_and_group,
    validate_document_visibility,
)
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility

# ===========================================================================
# ALLOWED_VISIBILITY_FOR_KIND constant tests
# ===========================================================================


class TestAllowedVisibilityForKind:
    def test_internal_allowed_visibilities(self):
        allowed = ALLOWED_VISIBILITY_FOR_KIND[UserKind.INTERNAL]
        assert DocumentVisibility.INTERNAL_PUBLIC in allowed
        assert DocumentVisibility.INTERNAL_GROUP in allowed
        assert DocumentVisibility.INTERNAL_PRIVATE in allowed
        assert DocumentVisibility.CLIENT_PRIVATE not in allowed

    def test_client_allowed_visibilities(self):
        allowed = ALLOWED_VISIBILITY_FOR_KIND[UserKind.CLIENT]
        assert DocumentVisibility.CLIENT_PRIVATE in allowed
        assert DocumentVisibility.INTERNAL_PUBLIC not in allowed
        assert DocumentVisibility.INTERNAL_GROUP not in allowed
        assert DocumentVisibility.INTERNAL_PRIVATE not in allowed


# ===========================================================================
# validate_document_visibility Tests
# ===========================================================================


class TestValidateDocumentVisibility:
    """Business rules:
    1. visibility must be in ALLOWED_VISIBILITY_FOR_KIND
    2. INTERNAL_PUBLIC requires admin role
    3. INTERNAL_GROUP requires group_id in user's groups
    """

    # --- Internal user happy paths ---

    def test_internal_admin_can_use_internal_public(self):
        validate_document_visibility(
            DocumentVisibility.INTERNAL_PUBLIC, None, UserKind.INTERNAL, UserRole.ADMIN, []
        )

    def test_internal_user_can_use_internal_private(self):
        validate_document_visibility(
            DocumentVisibility.INTERNAL_PRIVATE, None, UserKind.INTERNAL, UserRole.USER, []
        )

    def test_internal_user_can_use_internal_group(self):
        validate_document_visibility(
            DocumentVisibility.INTERNAL_GROUP, 5, UserKind.INTERNAL, UserRole.USER, [5]
        )

    # --- Client user happy paths ---

    def test_client_can_use_client_private(self):
        validate_document_visibility(
            DocumentVisibility.CLIENT_PRIVATE, None, UserKind.CLIENT, UserRole.USER, []
        )

    # --- Visibility not available for kind ---

    def test_client_cannot_use_internal_public(self):
        with pytest.raises(ValidationError, match="not available for kind"):
            validate_document_visibility(
                DocumentVisibility.INTERNAL_PUBLIC, None, UserKind.CLIENT, UserRole.USER, []
            )

    def test_client_cannot_use_internal_group(self):
        with pytest.raises(ValidationError, match="not available for kind"):
            validate_document_visibility(
                DocumentVisibility.INTERNAL_GROUP, 1, UserKind.CLIENT, UserRole.USER, []
            )

    def test_client_cannot_use_internal_private(self):
        with pytest.raises(ValidationError, match="not available for kind"):
            validate_document_visibility(
                DocumentVisibility.INTERNAL_PRIVATE, None, UserKind.CLIENT, UserRole.USER, []
            )

    def test_internal_cannot_use_client_private(self):
        with pytest.raises(ValidationError, match="not available for kind"):
            validate_document_visibility(
                DocumentVisibility.CLIENT_PRIVATE, None, UserKind.INTERNAL, UserRole.USER, []
            )

    # --- INTERNAL_PUBLIC requires admin ---

    def test_internal_user_cannot_use_internal_public(self):
        with pytest.raises(BusinessRuleViolation, match="Only admin can publish to internal_public"):
            validate_document_visibility(
                DocumentVisibility.INTERNAL_PUBLIC, None, UserKind.INTERNAL, UserRole.USER, []
            )

    # --- INTERNAL_GROUP requires group_id ---

    def test_internal_group_requires_group_id(self):
        with pytest.raises(ValidationError, match="group_id required"):
            validate_document_visibility(
                DocumentVisibility.INTERNAL_GROUP, None, UserKind.INTERNAL, UserRole.USER, []
            )

    def test_internal_group_requires_membership(self):
        with pytest.raises(BusinessRuleViolation, match="not a member of this group"):
            validate_document_visibility(
                DocumentVisibility.INTERNAL_GROUP, 5, UserKind.INTERNAL, UserRole.USER, [1, 2, 3]
            )

    def test_internal_group_with_valid_membership(self):
        validate_document_visibility(
            DocumentVisibility.INTERNAL_GROUP, 5, UserKind.INTERNAL, UserRole.USER, [5, 10]
        )


# ===========================================================================
# compute_owner_and_group Tests
# ===========================================================================


class TestComputeOwnerAndGroup:
    """Business rule: determine owner_id and group_id based on visibility."""

    def test_internal_public_returns_none_none(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PUBLIC, 5, 10)
        assert owner is None
        assert group is None

    def test_internal_group_returns_none_group_id(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_GROUP, 5, 10)
        assert owner is None
        assert group == 5

    def test_internal_private_returns_user_id_none(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PRIVATE, None, 10)
        assert owner == 10
        assert group is None

    def test_client_private_returns_user_id_none(self):
        owner, group = compute_owner_and_group(DocumentVisibility.CLIENT_PRIVATE, None, 10)
        assert owner == 10
        assert group is None

    def test_group_id_ignored_for_non_group_visibility(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PRIVATE, 99, 10)
        assert owner == 10
        assert group is None


# ===========================================================================
# can_view_document Tests
# ===========================================================================


class TestCanViewDocument:
    """Business rules for document viewing:
    - INTERNAL_PUBLIC: only internal users
    - INTERNAL_GROUP: internal users in the group
    - INTERNAL_PRIVATE: only the owner (internal)
    - CLIENT_PRIVATE: clients see their own; internal see assigned clients'
    """

    # --- INTERNAL_PUBLIC ---

    def test_internal_user_can_view_internal_public(self):
        assert can_view_document("internal_public", None, None, "internal", 1, [], []) is True

    def test_client_cannot_view_internal_public(self):
        assert can_view_document("internal_public", None, None, "client", 1, [], []) is False

    # --- INTERNAL_GROUP ---

    def test_internal_user_in_group_can_view(self):
        assert can_view_document("internal_group", None, 5, "internal", 1, [5, 10], []) is True

    def test_internal_user_not_in_group_cannot_view(self):
        assert can_view_document("internal_group", None, 5, "internal", 1, [1, 2], []) is False

    def test_client_cannot_view_internal_group(self):
        assert can_view_document("internal_group", None, 5, "client", 1, [], []) is False

    # --- INTERNAL_PRIVATE ---

    def test_owner_can_view_internal_private(self):
        assert can_view_document("internal_private", 10, None, "internal", 10, [], []) is True

    def test_non_owner_cannot_view_internal_private(self):
        assert can_view_document("internal_private", 10, None, "internal", 20, [], []) is False

    def test_client_cannot_view_internal_private(self):
        assert can_view_document("internal_private", 10, None, "client", 10, [], []) is False

    # --- CLIENT_PRIVATE ---

    def test_client_owner_can_view_client_private(self):
        assert can_view_document("client_private", 10, None, "client", 10, [], []) is True

    def test_client_non_owner_cannot_view_client_private(self):
        assert can_view_document("client_private", 10, None, "client", 20, [], []) is False

    def test_internal_assigned_can_view_client_private(self):
        assert can_view_document("client_private", 10, None, "internal", 1, [], [10]) is True

    def test_internal_not_assigned_cannot_view_client_private(self):
        assert can_view_document("client_private", 10, None, "internal", 1, [], [20]) is False

    def test_internal_no_assignments_cannot_view_client_private(self):
        assert can_view_document("client_private", 10, None, "internal", 1, [], []) is False

    # --- Edge cases ---

    def test_empty_group_ids_list(self):
        assert can_view_document("internal_group", None, 5, "internal", 1, [], []) is False

    def test_empty_assigned_client_ids_list(self):
        assert can_view_document("client_private", 10, None, "internal", 1, [], []) is False

    def test_client_with_groups_still_cannot_view_internal(self):
        assert can_view_document("internal_public", None, None, "client", 1, [1, 2, 3], []) is False


# ===========================================================================
# Parameterized tests for can_view_document
# ===========================================================================


class TestCanViewDocumentParameterized:
    """Parameterized matrix of visibility × user kind combinations."""

    @pytest.mark.parametrize(
        "visibility,owner_id,group_id,kind,user_id,groups,assigned,expected",
        [
            # INTERNAL_PUBLIC
            ("internal_public", None, None, "internal", 1, [], [], True),
            ("internal_public", None, None, "client", 1, [], [], False),
            # INTERNAL_GROUP
            ("internal_group", None, 5, "internal", 1, [5], [], True),
            ("internal_group", None, 5, "internal", 1, [3], [], False),
            ("internal_group", None, 5, "client", 1, [], [], False),
            # INTERNAL_PRIVATE
            ("internal_private", 10, None, "internal", 10, [], [], True),
            ("internal_private", 10, None, "internal", 20, [], [], False),
            ("internal_private", 10, None, "client", 10, [], [], False),
            # CLIENT_PRIVATE
            ("client_private", 10, None, "client", 10, [], [], True),
            ("client_private", 10, None, "client", 20, [], [], False),
            ("client_private", 10, None, "internal", 1, [], [10], True),
            ("client_private", 10, None, "internal", 1, [], [20], False),
        ],
    )
    def test_visibility_matrix(
        self, visibility, owner_id, group_id, kind, user_id, groups, assigned, expected
    ):
        result = can_view_document(visibility, owner_id, group_id, kind, user_id, groups, assigned)
        assert result is expected
