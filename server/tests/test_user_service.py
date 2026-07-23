"""
Tests for application/services/auth_service.py — authentication, user creation, toggle.
Tests the application service with mocked UoW factory.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from application.dto.auth_dto import CreateUserCommand, LoginCommand  # noqa: E402
from application.services.auth_service import AuthService  # noqa: E402
from domain.entities.user import User  # noqa: E402
from domain.exceptions import BusinessRuleViolation, ValidationError  # noqa: E402
from domain.value_objects.roles import UserKind, UserRole  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_service():
    hasher = MagicMock()
    hasher.hash.return_value = "hashed_pw"
    hasher.verify.return_value = True

    token_provider = MagicMock()
    token_provider.create_token.return_value = "token123"

    uow_factory = MagicMock()
    uow = MagicMock()
    uow_factory.create.return_value.__enter__ = MagicMock(return_value=uow)
    uow_factory.create.return_value.__exit__ = MagicMock(return_value=False)

    return AuthService(
        uow_factory=uow_factory,
        password_hasher=hasher,
        token_provider=token_provider,
    ), uow


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_successful_authentication(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="test@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        user.hashed_password = "hashed_pw"
        user.is_active = True
        uow.users.get_by_email.return_value = user

        result = service.authenticate(LoginCommand(email="test@test.com", password="password"))
        assert result.access_token == "token123"
        assert result.role == "user"

    def test_wrong_password_raises(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="test@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        user.hashed_password = "hashed_pw"
        user.is_active = True
        uow.users.get_by_email.return_value = user
        service._hasher.verify.return_value = False

        with pytest.raises(ValidationError):
            service.authenticate(LoginCommand(email="test@test.com", password="wrong"))


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


class TestCreateUser:
    def test_successful_creation(self, auth_service):
        service, uow = auth_service
        uow.users.get_by_email.return_value = None
        saved = User(id=1, email="new@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.save.return_value = saved

        result = service.create_user(
            CreateUserCommand(email="new@test.com", password="pw", role="user", kind="internal")
        )
        assert result.id == 1

    def test_invalid_role_raises(self, auth_service):
        service, uow = auth_service
        with pytest.raises(ValidationError):
            service.create_user(
                CreateUserCommand(email="x@x.com", password="pw", role="superadmin", kind="internal")
            )


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


class TestListUsers:
    def test_delegates_to_use_case(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="a@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.list_all.return_value = [user]

        result = service.list_users()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# toggle_active
# ---------------------------------------------------------------------------


class TestToggleActive:
    def test_deactivate_user(self, auth_service):
        service, uow = auth_service
        user = User(id=5, email="u@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.get_by_id.return_value = user

        result = service.toggle_active(user_id=5, is_active=False, admin_id=1)
        assert result["is_active"] is False

    def test_cannot_deactivate_self(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="admin@test.com", role=UserRole.ADMIN, kind=UserKind.INTERNAL)
        uow.users.get_by_id.return_value = user

        with pytest.raises(BusinessRuleViolation):
            service.toggle_active(user_id=1, is_active=False, admin_id=1)
