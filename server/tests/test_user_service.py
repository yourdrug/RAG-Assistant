"""
Tests for application/services/auth_service.py — authentication, user creation, toggle.
Tests the application service with mocked use cases.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from application.dto.auth_dto import CreateUserCommand, LoginCommand, LoginResult, UserDTO  # noqa: E402
from application.services.auth_service import AuthService  # noqa: E402
from domain.exceptions import BusinessRuleViolation, ValidationError  # noqa: E402
from fastapi import HTTPException  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_service():
    authenticate_user = MagicMock()
    create_user = MagicMock()
    list_users = MagicMock()
    toggle_user_active = MagicMock()
    return AuthService(
        authenticate_user=authenticate_user,
        create_user=create_user,
        list_users=list_users,
        toggle_user_active=toggle_user_active,
    )


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_successful_authentication(self, auth_service):
        expected = LoginResult(access_token="token123", role="user", kind="internal")
        auth_service._authenticate_user.execute.return_value = expected
        result = auth_service.authenticate(LoginCommand(email="test@test.com", password="password"))
        assert result.access_token == "token123"
        assert result.role == "user"

    def test_wrong_password_raises(self, auth_service):
        auth_service._authenticate_user.execute.side_effect = HTTPException(401, "Invalid credentials")
        with pytest.raises(HTTPException) as exc_info:
            auth_service.authenticate(LoginCommand(email="test@test.com", password="wrong"))
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


class TestCreateUser:
    def test_successful_creation(self, auth_service):
        expected = UserDTO(id=1, email="new@test.com", role="user", kind="internal", is_active=True)
        auth_service._create_user.execute.return_value = expected
        result = auth_service.create_user(
            CreateUserCommand(email="new@test.com", password="pw", role="user", kind="internal")
        )
        assert result.id == 1

    def test_invalid_role_raises(self, auth_service):
        auth_service._create_user.execute.side_effect = ValidationError("Invalid role")
        with pytest.raises(ValidationError):
            auth_service.create_user(
                CreateUserCommand(email="x@x.com", password="pw", role="superadmin", kind="internal")
            )


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


class TestListUsers:
    def test_delegates_to_use_case(self, auth_service):
        expected = [UserDTO(id=1, email="a@test.com", role="user", kind="internal", is_active=True)]
        auth_service._list_users.execute.return_value = expected
        result = auth_service.list_users()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# toggle_active
# ---------------------------------------------------------------------------


class TestToggleActive:
    def test_deactivate_user(self, auth_service):
        auth_service._toggle_user_active.execute.return_value = {"is_active": False}
        result = auth_service.toggle_active(user_id=5, is_active=False, admin_id=1)
        assert result["is_active"] is False

    def test_cannot_deactivate_self(self, auth_service):
        auth_service._toggle_user_active.execute.side_effect = BusinessRuleViolation(
            "Cannot deactivate yourself"
        )
        with pytest.raises(BusinessRuleViolation):
            auth_service.toggle_active(user_id=1, is_active=False, admin_id=1)
