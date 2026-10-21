"""Tests for identity authentication and user management.

Tests the application service with mocked UoW factory.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from application.identity import CreateUserCommand, LoginCommand  # noqa: E402
from application.identity import AuthService, UserManagementService  # noqa: E402
from domain.identity.user import User  # noqa: E402
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError  # noqa: E402
from domain.identity.roles import UserKind, UserRole  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_service():
    hasher = AsyncMock()
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


@pytest.fixture
def user_service(auth_service):
    service, uow = auth_service
    return UserManagementService(service._uow_factory, service._hasher), uow


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

    @pytest.mark.asyncio
    async def test_inactive_user_raises(self, auth_service):
        service, uow = auth_service
        user = User(id=1, email="inactive@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        user.hashed_password = "hashed_pw"
        user.is_active = False
        uow.users.get_by_email.return_value = user

        with pytest.raises(ValidationError, match="Invalid email or password"):
            await service.authenticate(LoginCommand(email="inactive@test.com", password="password"))
        service._hasher.verify.assert_awaited_once()
        password, dummy_hash = service._hasher.verify.await_args.args
        assert password == "password"
        assert dummy_hash.startswith("$2b$")
        service._tokens.create_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_email_raises(self, auth_service):
        service, uow = auth_service
        uow.users.get_by_email.return_value = None

        with pytest.raises(ValidationError, match="Invalid email or password"):
            await service.authenticate(LoginCommand(email="nobody@test.com", password="password"))
        service._hasher.verify.assert_awaited_once()
        password, dummy_hash = service._hasher.verify.await_args.args
        assert password == "password"
        assert dummy_hash.startswith("$2b$")
        service._tokens.create_token.assert_not_called()


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


class TestCreateUser:
    @pytest.mark.asyncio
    async def test_successful_creation(self, user_service):
        service, uow = user_service
        uow.users.get_by_email.return_value = None
        saved = User(id=1, email="new@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.save.return_value = saved

        result = await service.create_user(
            CreateUserCommand(email="new@test.com", password="pw", role="user", kind="internal")
        )
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_invalid_role_raises(self, user_service):
        service, uow = user_service
        with pytest.raises(ValidationError):
            await service.create_user(
                CreateUserCommand(email="x@x.com", password="pw", role="superadmin", kind="internal")
            )


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


class TestListUsers:
    @pytest.mark.asyncio
    async def test_delegates_to_use_case(self, user_service):
        service, uow = user_service
        user = User(id=1, email="a@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.list_all.return_value = [user]
        uow.users.count_all.return_value = 1

        users, total = await service.list_users()
        assert len(users) == 1
        assert total == 1


# ---------------------------------------------------------------------------
# toggle_active
# ---------------------------------------------------------------------------


class TestToggleActive:
    @pytest.mark.asyncio
    async def test_deactivate_user(self, user_service):
        service, uow = user_service
        user = User(id=5, email="u@test.com", role=UserRole.USER, kind=UserKind.INTERNAL)
        uow.users.get_by_id.return_value = user

        result = await service.toggle_active(user_id=5, is_active=False, admin_id=1)
        assert result.is_active is False

    @pytest.mark.asyncio
    async def test_cannot_deactivate_self(self, user_service):
        service, uow = user_service
        user = User(id=1, email="admin@test.com", role=UserRole.ADMIN, kind=UserKind.INTERNAL)
        uow.users.get_by_id.return_value = user

        with pytest.raises(BusinessRuleViolationError):
            await service.toggle_active(user_id=1, is_active=False, admin_id=1)
