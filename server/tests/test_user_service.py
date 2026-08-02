"""
Tests for application/services/auth_service.py — authentication, user creation, toggle.
Tests the application service with mocked UoW factory.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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
    uow = AsyncMock()
    uow_factory.create.return_value.__aenter__ = AsyncMock(return_value=uow)
    uow_factory.create.return_value.__aexit__ = AsyncMock(return_value=False)

    return AuthService(
        uow_factory=uow_factory,
        password_hasher=hasher,
        token_provider=token_provider,
    ), uow


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_successful_authentication(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="test@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        user.hashed_password = "hashed_pw"
        user.is_active = True
        uow.users.get_by_email.return_value = user

        result = await service.authenticate(LoginCommand(email="test@test.com", password="password"))
        assert result.access_token == "token123"
        assert result.role == "user"

    @pytest.mark.asyncio
    async def test_wrong_password_raises(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="test@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        user.hashed_password = "hashed_pw"
        user.is_active = True
        uow.users.get_by_email.return_value = user
        service._hasher.verify.return_value = False

        with pytest.raises(ValidationError):
            await service.authenticate(LoginCommand(email="test@test.com", password="wrong"))


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


class TestCreateUser:
    @pytest.mark.asyncio
    async def test_successful_creation(self, auth_service):
        service, uow = auth_service
        uow.users.get_by_email.return_value = None
        saved = User(id=1, email="new@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.save.return_value = saved

        result = await service.create_user(
            CreateUserCommand(email="new@test.com", password="pw", role="user", kind="internal")
        )
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_invalid_role_raises(self, auth_service):
        service, uow = auth_service
        with pytest.raises(ValidationError):
            await service.create_user(
                CreateUserCommand(email="x@x.com", password="pw", role="superadmin", kind="internal")
            )


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


class TestListUsers:
    @pytest.mark.asyncio
    async def test_delegates_to_use_case(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="a@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.list_all.return_value = [user]

        result = await service.list_users()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# toggle_active
# ---------------------------------------------------------------------------


class TestToggleActive:
    @pytest.mark.asyncio
    async def test_deactivate_user(self, auth_service):
        service, uow = auth_service
        user = User(id=5, email="u@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.get_by_id.return_value = user

        result = await service.toggle_active(user_id=5, is_active=False, admin_id=1)
        assert result["is_active"] is False

    @pytest.mark.asyncio
    async def test_cannot_deactivate_self(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="admin@test.com", role=UserRole.ADMIN, kind=UserKind.INTERNAL)
        uow.users.get_by_id.return_value = user

        with pytest.raises(BusinessRuleViolation):
            await service.toggle_active(user_id=1, is_active=False, admin_id=1)
