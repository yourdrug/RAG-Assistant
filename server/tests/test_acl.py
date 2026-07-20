"""
Tests for infrastructure/acl.py — access control logic.
Mocks database functions, tests pure business rules.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from fastapi import HTTPException  # noqa: E402
from infrastructure.acl import (  # noqa: E402
    ALLOWED_VISIBILITY_FOR_KIND,
    build_qdrant_filter,
    can_view_document,
    owner_and_group_for,
    validate_visibility,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _internal_user(user_id=1, role="user"):
    return {"id": user_id, "kind": "internal", "role": role}


def _client_user(user_id=100):
    return {"id": user_id, "kind": "client", "role": "user"}


def _doc(visibility="internal_public", owner_id=1, group_id=None):
    return {"visibility": visibility, "owner_id": owner_id, "group_id": group_id}


# ---------------------------------------------------------------------------
# ALLOWED_VISIBILITY_FOR_KIND
# ---------------------------------------------------------------------------


class TestAllowedVisibility:
    def test_internal_allowed_visibilities(self):
        assert ALLOWED_VISIBILITY_FOR_KIND["internal"] == {
            "internal_public",
            "internal_group",
            "internal_private",
        }

    def test_client_allowed_visibilities(self):
        assert ALLOWED_VISIBILITY_FOR_KIND["client"] == {"client_private"}


# ---------------------------------------------------------------------------
# validate_visibility
# ---------------------------------------------------------------------------


class TestValidateVisibility:
    def _mock_db_with_groups(self, group_ids):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=group_ids):
            yield db

    def test_internal_user_can_use_internal_private(self):
        db = MagicMock()
        validate_visibility("internal_private", None, _internal_user(), db)

    def test_internal_user_can_use_internal_group(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=[10]):
            validate_visibility("internal_group", 10, _internal_user(), db)

    def test_internal_user_cannot_use_client_private(self):
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            validate_visibility("client_private", None, _internal_user(), db)
        assert exc_info.value.status_code == 400

    def test_client_user_can_use_client_private(self):
        db = MagicMock()
        validate_visibility("client_private", None, _client_user(), db)

    def test_client_user_cannot_use_internal_public(self):
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            validate_visibility("internal_public", None, _client_user(), db)
        assert exc_info.value.status_code == 400

    def test_non_admin_cannot_publish_internal_public(self):
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            validate_visibility("internal_public", None, _internal_user(role="user"), db)
        assert exc_info.value.status_code == 403

    def test_admin_can_publish_internal_public(self):
        db = MagicMock()
        validate_visibility("internal_public", None, _internal_user(role="admin"), db)

    def test_internal_group_requires_group_id(self):
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            validate_visibility("internal_group", None, _internal_user(), db)
        assert exc_info.value.status_code == 400
        assert "group_id required" in exc_info.value.detail

    def test_internal_group_rejects_non_member(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=[1, 2]):
            with pytest.raises(HTTPException) as exc_info:
                validate_visibility("internal_group", 99, _internal_user(), db)
            assert exc_info.value.status_code == 403

    def test_invalid_visibility_value(self):
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            validate_visibility("nonexistent", None, _internal_user(), db)
        assert exc_info.value.status_code == 400


import pytest  # noqa: E402

# ---------------------------------------------------------------------------
# owner_and_group_for
# ---------------------------------------------------------------------------


class TestOwnerAndGroupFor:
    def test_internal_public_returns_none_none(self):
        owner, group = owner_and_group_for("internal_public", None, _internal_user())
        assert owner is None
        assert group is None

    def test_internal_group_returns_none_group_id(self):
        owner, group = owner_and_group_for("internal_group", 42, _internal_user())
        assert owner is None
        assert group == 42

    def test_internal_private_returns_owner(self):
        user = _internal_user(user_id=7)
        owner, group = owner_and_group_for("internal_private", None, user)
        assert owner == 7
        assert group is None

    def test_client_private_returns_owner(self):
        user = _client_user(user_id=99)
        owner, group = owner_and_group_for("client_private", None, user)
        assert owner == 99
        assert group is None


# ---------------------------------------------------------------------------
# can_view_document
# ---------------------------------------------------------------------------


class TestCanViewDocument:
    def test_internal_user_views_internal_public(self):
        db = MagicMock()
        assert can_view_document(db, _internal_user(), _doc("internal_public")) is True

    def test_client_user_rejected_from_internal_public(self):
        db = MagicMock()
        assert can_view_document(db, _client_user(), _doc("internal_public")) is False

    def test_internal_user_views_group_doc_if_member(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=[10]):
            assert can_view_document(db, _internal_user(), _doc("internal_group", group_id=10)) is True

    def test_internal_user_rejected_from_group_doc_if_not_member(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=[1, 2]):
            assert can_view_document(db, _internal_user(), _doc("internal_group", group_id=10)) is False

    def test_client_rejected_from_internal_group(self):
        db = MagicMock()
        assert can_view_document(db, _client_user(), _doc("internal_group", group_id=1)) is False

    def test_internal_owner_views_private_doc(self):
        db = MagicMock()
        assert can_view_document(db, _internal_user(user_id=5), _doc("internal_private", owner_id=5)) is True

    def test_internal_non_owner_rejected_from_private_doc(self):
        db = MagicMock()
        assert can_view_document(db, _internal_user(user_id=5), _doc("internal_private", owner_id=9)) is False

    def test_client_views_own_private_doc(self):
        db = MagicMock()
        assert can_view_document(db, _client_user(user_id=50), _doc("client_private", owner_id=50)) is True

    def test_client_rejected_from_other_client_doc(self):
        db = MagicMock()
        assert can_view_document(db, _client_user(user_id=50), _doc("client_private", owner_id=51)) is False

    def test_internal_views_assigned_client_doc(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_assigned_client_ids", return_value=[50, 51]):
            assert (
                can_view_document(db, _internal_user(user_id=1), _doc("client_private", owner_id=50)) is True
            )

    def test_internal_rejected_from_unassigned_client_doc(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_assigned_client_ids", return_value=[99]):
            assert (
                can_view_document(db, _internal_user(user_id=1), _doc("client_private", owner_id=50)) is False
            )

    def test_unknown_visibility_returns_false(self):
        db = MagicMock()
        assert can_view_document(db, _internal_user(), _doc("nonexistent")) is False


# ---------------------------------------------------------------------------
# build_qdrant_filter
# ---------------------------------------------------------------------------


class TestBuildQdrantFilter:
    def test_client_gets_owner_filter(self):
        db = MagicMock()
        user = _client_user(user_id=42)
        f = build_qdrant_filter(user, db)
        assert f.must is not None
        assert len(f.must) == 2

    def test_client_filter_has_correct_visibility(self):
        db = MagicMock()
        f = build_qdrant_filter(_client_user(user_id=1), db)
        vis_match = f.must[0]
        assert vis_match.match.value == "client_private"

    def test_internal_user_base_filter_has_public_and_private(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=[]):
            with patch("infrastructure.acl.get_assigned_client_ids", return_value=[]):
                f = build_qdrant_filter(_internal_user(user_id=1), db)
        assert f.should is not None
        assert len(f.should) == 2  # public + private

    def test_internal_with_groups_adds_group_filter(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=[10, 20]):
            with patch("infrastructure.acl.get_assigned_client_ids", return_value=[]):
                f = build_qdrant_filter(_internal_user(user_id=1), db)
        assert len(f.should) == 3  # public + private + group

    def test_internal_with_clients_adds_client_filter(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=[]):
            with patch("infrastructure.acl.get_assigned_client_ids", return_value=[50]):
                f = build_qdrant_filter(_internal_user(user_id=1), db)
        assert len(f.should) == 3  # public + private + client

    def test_internal_with_both_groups_and_clients(self):
        db = MagicMock()
        with patch("infrastructure.acl.get_user_group_ids", return_value=[10]):
            with patch("infrastructure.acl.get_assigned_client_ids", return_value=[50]):
                f = build_qdrant_filter(_internal_user(user_id=1), db)
        assert len(f.should) == 4  # public + private + group + client
