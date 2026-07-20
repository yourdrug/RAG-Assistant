"""
Tests for services/user_service.py — authentication, user creation, toggle.
Mocks database layer, tests pure business validation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from services.user_service import UserService  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    return UserService()


def _mock_db():
    return MagicMock()


def _fake_user(
    user_id=1, email="test@test.com", role="user", kind="internal", is_active=True, hashed_password="hash"
):
    return {
        "id": user_id,
        "email": email,
        "role": role,
        "kind": kind,
        "is_active": is_active,
        "hashed_password": hashed_password,
    }


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_successful_authentication(self, service):
        db = _mock_db()
        user = _fake_user()
        with patch("services.user_service.get_user_by_email", return_value=user):
            with patch("services.user_service.verify_password", return_value=True):
                with patch("services.user_service.create_access_token", return_value="token123"):
                    result = service.authenticate("test@test.com", "password", db)
        assert result["access_token"] == "token123"
        assert result["role"] == "user"
        assert result["kind"] == "internal"

    def test_wrong_password_raises_401(self, service):
        db = _mock_db()
        user = _fake_user()
        with patch("services.user_service.get_user_by_email", return_value=user):
            with patch("services.user_service.verify_password", return_value=False):
                with pytest.raises(HTTPException) as exc_info:
                    service.authenticate("test@test.com", "wrong", db)
        assert exc_info.value.status_code == 401

    def test_nonexistent_user_raises_401(self, service):
        db = _mock_db()
        with patch("services.user_service.get_user_by_email", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                service.authenticate("nobody@test.com", "pw", db)
        assert exc_info.value.status_code == 401

    def test_inactive_user_raises_401(self, service):
        db = _mock_db()
        user = _fake_user(is_active=False)
        with patch("services.user_service.get_user_by_email", return_value=user):
            with pytest.raises(HTTPException) as exc_info:
                service.authenticate("test@test.com", "pw", db)
        assert exc_info.value.status_code == 401

    def test_error_message_does_not_leak_info(self, service):
        db = _mock_db()
        with patch("services.user_service.get_user_by_email", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                service.authenticate("x@x.com", "y", db)
        assert "not found" not in exc_info.value.detail.lower()
        assert "invalid" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


class TestCreateUser:
    def test_successful_creation(self, service):
        db = _mock_db()
        with patch("services.user_service.get_user_by_email", return_value=None):
            with patch("services.user_service.create_user", return_value={"id": 1}):
                with patch("services.user_service.hash_password", return_value="hashed"):
                    result = service.create_user("new@test.com", "pw", "user", "internal", db)
        assert result["id"] == 1

    def test_invalid_role_raises_400(self, service):
        db = _mock_db()
        with pytest.raises(HTTPException) as exc_info:
            service.create_user("x@x.com", "pw", "superadmin", "internal", db)
        assert exc_info.value.status_code == 400
        assert "role" in exc_info.value.detail.lower()

    def test_invalid_kind_raises_400(self, service):
        db = _mock_db()
        with pytest.raises(HTTPException) as exc_info:
            service.create_user("x@x.com", "pw", "user", "external", db)
        assert exc_info.value.status_code == 400
        assert "kind" in exc_info.value.detail.lower()

    def test_client_cannot_be_admin(self, service):
        db = _mock_db()
        with pytest.raises(HTTPException) as exc_info:
            service.create_user("x@x.com", "pw", "admin", "client", db)
        assert exc_info.value.status_code == 400
        assert "client" in exc_info.value.detail.lower()

    def test_duplicate_email_raises_409(self, service):
        db = _mock_db()
        existing = _fake_user(email="taken@test.com")
        with patch("services.user_service.get_user_by_email", return_value=existing):
            with pytest.raises(HTTPException) as exc_info:
                service.create_user("taken@test.com", "pw", "user", "internal", db)
        assert exc_info.value.status_code == 409

    def test_admin_role_accepted(self, service):
        db = _mock_db()
        with patch("services.user_service.get_user_by_email", return_value=None):
            with patch("services.user_service.create_user", return_value={"id": 2}):
                with patch("services.user_service.hash_password", return_value="h"):
                    result = service.create_user("admin@test.com", "pw", "admin", "internal", db)
        assert result["id"] == 2

    def test_client_role_accepted(self, service):
        db = _mock_db()
        with patch("services.user_service.get_user_by_email", return_value=None):
            with patch("services.user_service.create_user", return_value={"id": 3}):
                with patch("services.user_service.hash_password", return_value="h"):
                    result = service.create_user("client@test.com", "pw", "user", "client", db)
        assert result["id"] == 3


# ---------------------------------------------------------------------------
# toggle_active
# ---------------------------------------------------------------------------


class TestToggleActive:
    def test_deactivate_other_user(self, service):
        db = _mock_db()
        with patch("services.user_service.set_user_active", return_value=True):
            result = service.toggle_active(user_id=5, is_active=False, admin_id=1, db=db)
        assert result["is_active"] is False

    def test_activate_user(self, service):
        db = _mock_db()
        with patch("services.user_service.set_user_active", return_value=True):
            result = service.toggle_active(user_id=5, is_active=True, admin_id=1, db=db)
        assert result["is_active"] is True

    def test_cannot_deactivate_self(self, service):
        db = _mock_db()
        with pytest.raises(HTTPException) as exc_info:
            service.toggle_active(user_id=1, is_active=False, admin_id=1, db=db)
        assert exc_info.value.status_code == 400
        assert "yourself" in exc_info.value.detail.lower()

    def test_can_activate_self(self, service):
        db = _mock_db()
        with patch("services.user_service.set_user_active", return_value=True):
            result = service.toggle_active(user_id=1, is_active=True, admin_id=1, db=db)
        assert result["is_active"] is True

    def test_nonexistent_user_raises_404(self, service):
        db = _mock_db()
        with patch("services.user_service.set_user_active", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                service.toggle_active(user_id=999, is_active=False, admin_id=1, db=db)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


class TestListUsers:
    def test_delegates_to_database(self, service):
        db = _mock_db()
        expected = [_fake_user(), _fake_user(user_id=2)]
        with patch("services.user_service.list_users", return_value=expected):
            result = service.list_users(db)
        assert len(result) == 2
