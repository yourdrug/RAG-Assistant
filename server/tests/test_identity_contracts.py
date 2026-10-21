"""Identity compatibility, scope loading and transaction ordering."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from application.identity import (
    ApiKeyService,
    AssignmentService,
    ChangeRoleCommand,
    GroupService,
    UserContextFactory,
    UserManagementService,
)
from domain.identity.roles import UserKind, UserRole
from domain.identity.user import User
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError, ValidationError


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.identity", "application.identity.assignment_service", "AssignmentService"),
        ("application.identity", "application.identity.auth_service", "AuthService"),
        ("application.identity", "application.identity.group_service", "GroupService"),
        ("application.identity", "application.identity.user_context_factory", "UserContextFactory"),
        ("application.identity", "application.identity.ports", "ApiKeyProviderPort"),
        ("application.identity", "application.identity.dto", "ApiKeyAuthResult"),
        ("application.identity", "application.identity.dto", "ApiKeyInfo"),
        ("application.identity", "application.identity.dto", "ChangeRoleCommand"),
        ("application.identity", "application.identity.dto", "CreateUserCommand"),
        ("application.identity", "application.identity.dto", "IssueApiKeyResult"),
        ("application.identity", "application.identity.dto", "LoginCommand"),
        ("application.identity", "application.identity.dto", "LoginResult"),
        ("application.identity", "application.identity.dto", "ToggleActiveResult"),
        ("application.identity", "application.identity.dto", "UserDTO"),
    ],
)
def test_public_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


def transaction() -> tuple[Mock, AsyncMock]:
    factory = Mock()
    uow = AsyncMock()
    factory.create.return_value.__aenter__ = AsyncMock(return_value=uow)
    factory.create.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory, uow


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [UserRole.USER, UserRole.ADMIN, UserRole.CURATOR])
async def test_context_loads_assignments_only_for_curator(role: UserRole) -> None:
    uow = AsyncMock()
    uow.groups.get_user_group_ids.return_value = [2]
    uow.assignments.get_managed_client_ids.return_value = [10]
    uow.assignments.get_managed_internal_ids.return_value = [11]
    uow.assignments.get_managed_group_ids.return_value = [12]

    ctx = await UserContextFactory().build(uow, 1, UserKind.INTERNAL, role)

    assert ctx.group_ids == (2,)

    if role == UserRole.CURATOR:
        assert ctx.managed_client_ids == (10,)
        assert ctx.managed_internal_ids == (11,)
        assert ctx.managed_group_ids == (12,)
    else:
        assert ctx.managed_client_ids == ctx.managed_internal_ids == ctx.managed_group_ids == ()
        uow.assignments.get_managed_client_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_context_does_not_load_group_membership() -> None:
    uow = AsyncMock()
    ctx = await UserContextFactory().build(uow, 1, UserKind.CLIENT)
    assert ctx.group_ids == ()
    uow.groups.get_user_group_ids.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,kind,expected",
    [
        (UserRole.ADMIN, UserKind.INTERNAL, "all"),
        (UserRole.USER, UserKind.CLIENT, "none"),
        (UserRole.USER, UserKind.INTERNAL, "own"),
        (UserRole.CURATOR, UserKind.INTERNAL, "scope"),
    ],
)
async def test_group_listing_preserves_scope(role: UserRole, kind: UserKind, expected: str) -> None:
    factory, uow = transaction()
    uow.groups.get_user_group_ids.return_value = [2]
    uow.assignments.get_managed_group_ids.return_value = [2, 3]
    uow.assignments.get_managed_client_ids.return_value = []
    uow.assignments.get_managed_internal_ids.return_value = []
    service = GroupService(factory)

    result = await service.list_for_user(1, role, kind)

    if expected == "all":
        assert result is uow.groups.list_all.return_value
        uow.groups.list_by_ids.assert_not_awaited()
    elif expected == "none":
        assert result == []
        uow.groups.list_all.assert_not_awaited()
        uow.groups.list_by_ids.assert_not_awaited()
    else:
        ids = uow.groups.list_by_ids.await_args.args[0]
        assert set(ids) == ({2, 3} if expected == "scope" else {2})
        uow.groups.list_all.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [UserKind.INTERNAL, UserKind.CLIENT])
async def test_group_membership_accepts_only_internal_users(kind: UserKind) -> None:
    factory, uow = transaction()
    uow.users.get_by_id.return_value = User(id=1, email="member@example.test", kind=kind)

    if kind == UserKind.CLIENT:
        with pytest.raises(ValidationError):
            await GroupService(factory).add_member(2, 1)
        uow.groups.add_user.assert_not_awaited()
    else:
        await GroupService(factory).add_member(2, 1)
        uow.groups.add_user.assert_awaited_once_with(1, 2)


@pytest.mark.asyncio
async def test_downgrading_curator_clears_assignments() -> None:
    factory, uow = transaction()
    uow.users.get_by_id.return_value = User(id=1, email="curator@example.test", role=UserRole.CURATOR)
    service = UserManagementService(factory, AsyncMock())

    result = await service.change_role(ChangeRoleCommand(1, UserRole.USER), admin_id=2)

    assert result.role == UserRole.USER
    uow.assignments.clear_all_for_manager.assert_awaited_once_with(1)
    uow.users.update_role.assert_awaited_once_with(1, UserRole.USER)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "missing", "commit_failure"])
async def test_api_key_invalidation_requires_successful_commit(outcome: str) -> None:
    factory, uow = transaction()
    provider = AsyncMock()
    service = ApiKeyService(factory, provider)
    events = []

    async def exit_transaction(*_args: object) -> bool:
        events.append("commit")

        if outcome == "commit_failure":
            msg = "commit failed"
            raise RuntimeError(msg)

        return False

    async def invalidate(key_id: int) -> None:
        events.append("invalidate")
        assert key_id == 7

    factory.create.return_value.__aexit__.side_effect = exit_transaction
    provider.invalidate_by_id.side_effect = invalidate
    uow.api_keys.revoke.return_value = outcome != "missing"

    if outcome == "success":
        await service.revoke_api_key(7, client_user_id=1)
        assert events == ["commit", "invalidate"]
    else:
        with pytest.raises(EntityNotFoundError if outcome == "missing" else RuntimeError):
            await service.revoke_api_key(7, client_user_id=1)
        provider.invalidate_by_id.assert_not_awaited()

    uow.api_keys.revoke.assert_awaited_once_with(7, user_id=1)


@pytest.mark.asyncio
async def test_listing_api_key_without_persisted_id_fails() -> None:
    factory, uow = transaction()
    uow.api_keys.list_for_user.return_value = [SimpleNamespace(id=None)]
    service = ApiKeyService(factory, AsyncMock())
    with pytest.raises(RuntimeError, match="ApiKey id is None"):
        await service.list_api_keys(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [UserRole.USER, UserRole.ADMIN, UserRole.CURATOR])
async def test_assign_user_checks_target_role(role: UserRole) -> None:
    factory, uow = transaction()
    uow.users.get_by_id.side_effect = [
        User(id=1, email="curator@example.test", role=UserRole.CURATOR),
        User(id=2, email="target@example.test", role=role),
    ]
    service = AssignmentService(factory)

    if role == UserRole.USER:
        await service.assign_user(1, 2, 3)
        uow.assignments.assign_user.assert_awaited_once_with(1, 2, 3)
    else:
        with pytest.raises(BusinessRuleViolationError):
            await service.assign_user(1, 2, 3)
        uow.assignments.assign_user.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_role", [UserRole.USER, UserRole.CURATOR])
@pytest.mark.parametrize("group_exists", [True, False])
async def test_assign_group_checks_manager_and_group(manager_role: UserRole, *, group_exists: bool) -> None:
    factory, uow = transaction()
    uow.users.get_by_id.return_value = User(id=1, email="manager@example.test", role=manager_role)
    uow.groups.list_by_ids.return_value = [SimpleNamespace(id=2)] if group_exists else []
    service = AssignmentService(factory)

    if manager_role != UserRole.CURATOR:
        with pytest.raises(BusinessRuleViolationError):
            await service.assign_group(1, 2, 3)
        uow.groups.list_by_ids.assert_not_awaited()
    elif not group_exists:
        with pytest.raises(EntityNotFoundError):
            await service.assign_group(1, 2, 3)
    else:
        await service.assign_group(1, 2, 3)
        uow.assignments.assign_group.assert_awaited_once_with(1, 2, 3)
        return

    uow.assignments.assign_group.assert_not_awaited()
