"""Regression checks for the separated identity services and HTTP wiring."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
import httpx
import pytest
import pytest_asyncio

from application.identity import (
    ApiKeyAuthResult,
    ApiKeyService,
    AuthService,
    CreateUserCommand,
    UserManagementService,
)
from domain.identity.api_key import ApiKey
from domain.identity.ports import TokenClaims
from domain.identity.roles import UserKind, UserRole
from domain.identity.user import User
from domain.shared.exceptions import AppClientError, BusinessRuleViolationError, EntityNotFoundError
from infrastructure.logging.log_buffer import LogBufferHandler
from presentation.api.dependencies import create_action_logger, create_log_buffer, create_rate_limiter
from presentation.api.exception_handlers import handle_client_exception
from presentation.api.routes.admin_logs import router as admin_logs_router
from presentation.api.routes.api_keys import router as api_key_router
from presentation.api.routes.auth import router as auth_router


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any


@pytest.fixture
def identity() -> SimpleNamespace:
    uow = AsyncMock()
    factory = Mock()
    factory.create.return_value.__aenter__ = AsyncMock(return_value=uow)
    factory.create.return_value.__aexit__ = AsyncMock(return_value=False)
    hasher = AsyncMock()
    tokens = Mock()
    tokens.create_token.return_value = "issued-token"
    tokens.decode_token.return_value = TokenClaims(user_id=1, role="user", expires_at=datetime.now(UTC))
    provider = Mock()
    provider.MISS = object()
    provider.get_cached = AsyncMock(return_value=provider.MISS)
    provider.set_cached = AsyncMock()
    provider.invalidate_by_id = AsyncMock()
    return SimpleNamespace(
        uow=uow,
        factory=factory,
        provider=provider,
        log_buffer=LogBufferHandler(),
        action_logger=Mock(),
        application=SimpleNamespace(
            auth_service=AuthService(factory, hasher, tokens),
            user_management_service=UserManagementService(factory, hasher),
            api_key_service=ApiKeyService(factory, provider),
        ),
        infrastructure=SimpleNamespace(services=SimpleNamespace(api_key_provider=provider)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [UserKind.INTERNAL, UserKind.CLIENT, None])
async def test_issue_key_only_for_existing_client(identity: SimpleNamespace, kind: UserKind | None) -> None:
    identity.uow.users.get_by_id.return_value = User(id=1, kind=kind) if kind else None
    identity.uow.api_keys.create.return_value = ApiKey(id=7, key_prefix="prefix")

    if kind == UserKind.CLIENT:
        result = await identity.application.api_key_service.issue_api_key(1, name="integration")
        assert result.id == 7
        identity.uow.api_keys.create.assert_awaited_once_with(
            user_id=1,
            key_hash=identity.provider.hash_key.return_value,
            key_prefix=identity.provider.key_prefix_for_display.return_value,
            name="integration",
        )
    else:
        with pytest.raises(EntityNotFoundError if kind is None else BusinessRuleViolationError):
            await identity.application.api_key_service.issue_api_key(1)
        identity.uow.api_keys.create.assert_not_awaited()
        identity.provider.generate_key.assert_not_called()

    identity.factory.create.assert_called_once_with(master=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [UserRole.USER, UserRole.CURATOR])
async def test_user_creation_rejects_non_admin(identity: SimpleNamespace, role: UserRole) -> None:
    with pytest.raises(BusinessRuleViolationError, match="Only admin"):
        await identity.application.user_management_service.create_user(
            CreateUserCommand("user@example.test", "password"), creator_role=role
        )
    identity.uow.users.save.assert_not_awaited()


@pytest_asyncio.fixture
async def client(identity: SimpleNamespace) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(api_key_router)
    app.include_router(admin_logs_router)
    app.add_exception_handler(AppClientError, handle_client_exception)
    app.state.container = identity
    app.dependency_overrides[create_rate_limiter] = lambda: None

    app.dependency_overrides[create_action_logger] = lambda: identity.action_logger
    app.dependency_overrides[create_log_buffer] = lambda: identity.log_buffer
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=cast("Any", app)), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_login_audits_authenticated_user_id(
    client: httpx.AsyncClient, identity: SimpleNamespace
) -> None:
    identity.uow.users.get_by_email.return_value = User(id=7, email="actor@example.com", is_active=True)

    response = await client.post("/auth/login", json={"email": "actor@example.com", "password": "password"})

    assert response.status_code == 200
    assert set(response.json()) == {"access_token", "token_type"}
    identity.action_logger.assert_called_once_with("login", user_id=7)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", list(UserRole))
async def test_action_logs_expose_actor_only_to_admin(
    client: httpx.AsyncClient, identity: SimpleNamespace, role: UserRole
) -> None:
    identity.uow.users.get_by_id.return_value = User(id=1, role=role, is_active=True)
    identity.uow.users.get_by_ids.return_value = [User(id=7, email="actor@example.com")]

    for message, actor_id in [
        ("group.add_member | user=7 | user_id=42", 7),
        ("login | user=7", 7),
        ("login | user=9", 9),
        ("system action", None),
    ]:
        identity.log_buffer.emit(
            logging.makeLogRecord(
                {"name": "actions", "levelname": "INFO", "msg": message, "action_user_id": actor_id}
            )
        )

    response = await client.get("/admin/logs", headers={"Authorization": "Bearer token"})

    if role == UserRole.ADMIN:
        assert response.status_code == 200
        assert [(log["user_id"], log["user_login"]) for log in response.json()["logs"]] == [
            (None, None),
            (9, None),
            (7, "actor@example.com"),
            (7, "actor@example.com"),
        ]
        identity.uow.users.get_by_ids.assert_awaited_once_with([7, 9])
    else:
        assert response.status_code == 403
        identity.uow.users.get_by_ids.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [True, False])
async def test_jwt_loads_current_user(
    client: httpx.AsyncClient, identity: SimpleNamespace, *, active: bool
) -> None:
    identity.uow.users.get_by_id.return_value = User(id=1, is_active=active)
    response = await client.get("/auth/me", headers={"Authorization": "Bearer token"})
    assert response.status_code == (200 if active else 401)
    identity.uow.users.get_by_id.assert_awaited_once_with(1)
    identity.uow.api_keys.get_active_client_by_hash.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("valid", [True, False])
async def test_api_key_authentication_preserves_cache_and_touch(
    client: httpx.AsyncClient, identity: SimpleNamespace, *, cached: bool, valid: bool
) -> None:
    result = (
        ApiKeyAuthResult(7, 1, "client@example.test", UserRole.USER, UserKind.CLIENT, True) if valid else None
    )
    identity.uow.api_keys.get_active_client_by_hash.return_value = result

    cache_payload = asdict(result) if result is not None else None

    if cached:
        identity.provider.get_cached.return_value = cache_payload

    response = await client.get("/auth/me", headers={"Authorization": "Api-Key secret"})
    assert response.status_code == (200 if valid else 401)

    if cached:
        identity.uow.api_keys.get_active_client_by_hash.assert_not_awaited()
        identity.provider.set_cached.assert_not_awaited()
    else:
        identity.uow.api_keys.get_active_client_by_hash.assert_awaited_once_with(
            identity.provider.hash_key.return_value
        )
        identity.provider.set_cached.assert_awaited_once_with(
            identity.provider.hash_key.return_value, cache_payload
        )

    if valid and not cached:
        identity.uow.api_keys.touch_last_used.assert_awaited_once_with(7)
    else:
        identity.uow.api_keys.touch_last_used.assert_not_awaited()

    identity.uow.users.get_by_id.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [UserRole.USER, UserRole.ADMIN])
async def test_user_listing_uses_management_service_and_admin_check(
    client: httpx.AsyncClient, identity: SimpleNamespace, role: UserRole
) -> None:
    identity.uow.users.get_by_id.return_value = User(id=1, role=role)
    identity.uow.users.list_all.return_value = []
    identity.uow.users.count_all.return_value = 0
    response = await client.get("/auth/users", headers={"Authorization": "Bearer token"})
    assert response.status_code == (200 if role == UserRole.ADMIN else 403)

    if role == UserRole.ADMIN:
        identity.uow.users.list_all.assert_awaited_once_with(limit=100, offset=0)
    else:
        identity.uow.users.list_all.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_id", [1, 2])
async def test_api_key_listing_checks_owner(
    client: httpx.AsyncClient, identity: SimpleNamespace, target_id: int
) -> None:
    identity.uow.users.get_by_id.return_value = User(id=1, kind=UserKind.CLIENT)
    identity.uow.api_keys.list_for_user.return_value = []
    response = await client.get(f"/clients/{target_id}/api-keys", headers={"Authorization": "Bearer token"})
    assert response.status_code == (200 if target_id == 1 else 403)

    if target_id == 1:
        identity.uow.api_keys.list_for_user.assert_awaited_once_with(1)
    else:
        identity.uow.api_keys.list_for_user.assert_not_awaited()
