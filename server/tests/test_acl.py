"""
Tests for access control logic.
- Domain rules: domain/services/access_control.py
- Qdrant filter: infrastructure/acl.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from domain.services.access_control import (  # noqa: E402
    ALLOWED_VISIBILITY_FOR_KIND,
    can_view_document,
    compute_owner_and_group,
    validate_document_visibility,
)
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.acl import build_qdrant_filter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _internal_user(user_id=1, role=UserRole.USER):
    return user_id, UserKind.INTERNAL, role


def _client_user(user_id=100):
    return user_id, UserKind.CLIENT, UserRole.USER


# ---------------------------------------------------------------------------
# ALLOWED_VISIBILITY_FOR_KIND
# ---------------------------------------------------------------------------


class TestAllowedVisibility:
    def test_internal_allowed_visibilities(self):
        assert ALLOWED_VISIBILITY_FOR_KIND[UserKind.INTERNAL] == {
            DocumentVisibility.INTERNAL_PUBLIC,
            DocumentVisibility.INTERNAL_GROUP,
            DocumentVisibility.INTERNAL_PRIVATE,
        }

    def test_client_allowed_visibilities(self):
        assert ALLOWED_VISIBILITY_FOR_KIND[UserKind.CLIENT] == {DocumentVisibility.CLIENT_PRIVATE}


# ---------------------------------------------------------------------------
# validate_document_visibility
# ---------------------------------------------------------------------------


class TestValidateVisibility:
    def test_internal_user_can_use_internal_private(self):
        uid, kind, role = _internal_user()
        validate_document_visibility(DocumentVisibility.INTERNAL_PRIVATE, None, kind, role, [])

    def test_internal_user_can_use_internal_group(self):
        uid, kind, role = _internal_user()
        validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, 10, kind, role, [10])

    def test_internal_user_cannot_use_client_private(self):
        uid, kind, role = _internal_user()
        with pytest.raises(Exception):
            validate_document_visibility(DocumentVisibility.CLIENT_PRIVATE, None, kind, role, [])

    def test_client_user_can_use_client_private(self):
        uid, kind, role = _client_user()
        validate_document_visibility(DocumentVisibility.CLIENT_PRIVATE, None, kind, role, [])

    def test_client_user_cannot_use_internal_public(self):
        uid, kind, role = _client_user()
        with pytest.raises(Exception):
            validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, kind, role, [])

    def test_non_admin_cannot_publish_internal_public(self):
        uid, kind, role = _internal_user(role=UserRole.USER)
        with pytest.raises(Exception):
            validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, kind, role, [])

    def test_admin_can_publish_internal_public(self):
        uid, kind, role = _internal_user(role=UserRole.ADMIN)
        validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, kind, role, [])

    def test_internal_group_requires_group_id(self):
        uid, kind, role = _internal_user()
        with pytest.raises(Exception):
            validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, None, kind, role, [])

    def test_internal_group_rejects_non_member(self):
        uid, kind, role = _internal_user()
        with pytest.raises(Exception):
            validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, 99, kind, role, [1, 2])


# ---------------------------------------------------------------------------
# compute_owner_and_group
# ---------------------------------------------------------------------------


class TestOwnerAndGroup:
    def test_internal_public_returns_none_none(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PUBLIC, None, 1)
        assert owner is None
        assert group is None

    def test_internal_group_returns_none_group_id(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_GROUP, 42, 1)
        assert owner is None
        assert group == 42

    def test_internal_private_returns_owner(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PRIVATE, None, 7)
        assert owner == 7
        assert group is None

    def test_client_private_returns_owner(self):
        owner, group = compute_owner_and_group(DocumentVisibility.CLIENT_PRIVATE, None, 99)
        assert owner == 99
        assert group is None


# ---------------------------------------------------------------------------
# can_view_document
# ---------------------------------------------------------------------------


class TestCanViewDocument:
    def test_internal_user_views_internal_public(self):
        assert can_view_document("internal_public", 1, None, "internal", 1, [], []) is True

    def test_client_user_rejected_from_internal_public(self):
        assert can_view_document("internal_public", 1, None, "client", 100, [], []) is False

    def test_internal_user_views_group_doc_if_member(self):
        assert can_view_document("internal_group", None, 10, "internal", 1, [10], []) is True

    def test_internal_user_rejected_from_group_doc_if_not_member(self):
        assert can_view_document("internal_group", None, 10, "internal", 1, [1, 2], []) is False

    def test_client_rejected_from_internal_group(self):
        assert can_view_document("internal_group", None, 1, "client", 100, [], []) is False

    def test_internal_owner_views_private_doc(self):
        assert can_view_document("internal_private", 5, None, "internal", 5, [], []) is True

    def test_internal_non_owner_rejected_from_private_doc(self):
        assert can_view_document("internal_private", 5, None, "internal", 9, [], []) is False

    def test_client_views_own_private_doc(self):
        assert can_view_document("client_private", 50, None, "client", 50, [], []) is True

    def test_client_rejected_from_other_client_doc(self):
        assert can_view_document("client_private", 51, None, "client", 50, [], []) is False

    def test_internal_views_assigned_client_doc(self):
        assert can_view_document("client_private", 50, None, "internal", 1, [], [50, 51]) is True

    def test_internal_rejected_from_unassigned_client_doc(self):
        assert can_view_document("client_private", 50, None, "internal", 1, [], [99]) is False

    def test_unknown_visibility_raises(self):
        with pytest.raises(ValueError):
            can_view_document("nonexistent", 1, None, "internal", 1, [], [])


# ---------------------------------------------------------------------------
# build_qdrant_filter (infrastructure layer)
# ---------------------------------------------------------------------------


class TestBuildQdrantFilter:
    def test_client_gets_owner_filter(self):
        user = {"id": 42, "kind": "client"}
        f = build_qdrant_filter(user, [], [])
        # Single condition: must=[visibility, owner_id] wrapped in should
        assert f.should is not None
        assert len(f.should) == 1
        inner = f.should[0]
        assert inner.must is not None
        assert len(inner.must) == 2

    def test_client_filter_has_correct_visibility(self):
        f = build_qdrant_filter({"id": 1, "kind": "client"}, [], [])
        inner = f.should[0]
        vis_match = inner.must[0]
        assert vis_match.match.value == "client_private"

    def test_internal_user_base_filter_has_public_and_private(self):
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [], [])
        assert f.should is not None
        assert len(f.should) == 2  # public + private

    def test_internal_with_groups_adds_group_filter(self):
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [10, 20], [])
        assert len(f.should) == 3  # public + private + group

    def test_internal_with_clients_adds_client_filter(self):
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [], [50])
        assert len(f.should) == 3  # public + private + client

    def test_internal_with_both_groups_and_clients(self):
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [10], [50])
        assert len(f.should) == 4  # public + private + group + client
