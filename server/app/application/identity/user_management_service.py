"""User administration with domain validation and transactional role changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.identity.dto import ToggleActiveResult, UserDTO
from domain.identity.roles import UserKind, UserRole
from domain.identity.user import User
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError


if TYPE_CHECKING:
    from application.identity.dto import ChangeRoleCommand, CreateUserCommand
    from application.uow import UnitOfWorkFactory
    from domain.identity.ports import PasswordHasher


class UserManagementService:
    def __init__(self, uow_factory: UnitOfWorkFactory, password_hasher: PasswordHasher) -> None:
        self._uow_factory = uow_factory
        self._hasher = password_hasher

    async def create_user(
        self, command: CreateUserCommand, creator_role: str | UserRole = UserRole.ADMIN
    ) -> UserDTO:
        async with self._uow_factory.create(master=True) as uow:
            role = UserRole.validate(command.role)
            kind = UserKind.validate(command.kind)

            user = User(
                email=command.email,
                role=role,
                kind=kind,
            )
            user.ensure_valid_for_creation()
            user.ensure_can_be_created_by(UserRole(creator_role))

            if await uow.users.get_by_email(command.email) is not None:
                msg = "User with this email already exists"
                raise BusinessRuleViolationError(msg)

            user.hashed_password = await self._hasher.hash(command.password)
            saved = await uow.users.save(user)

            if saved.id is None:
                msg = "User save returned None id"
                raise RuntimeError(msg)

            return UserDTO(
                id=saved.id,
                email=saved.email,
                role=saved.role,
                kind=saved.kind,
                is_active=saved.is_active,
            )

    async def list_users(self, limit: int = 100, offset: int = 0) -> tuple[list[UserDTO], int]:
        async with self._uow_factory.create() as uow:
            users = await uow.users.list_all(limit=limit, offset=offset)
            total = await uow.users.count_all()
            return (
                [
                    UserDTO(
                        id=u.id if u.id is not None else 0,
                        email=u.email,
                        role=u.role,
                        kind=u.kind,
                        is_active=u.is_active,
                    )
                    for u in users
                ],
                total,
            )

    async def toggle_active(self, user_id: int, *, is_active: bool, admin_id: int) -> ToggleActiveResult:
        async with self._uow_factory.create(master=True) as uow:
            user = await uow.users.get_by_id(user_id)

            if user is None:
                msg = "User"
                raise EntityNotFoundError(msg, user_id)

            user.deactivate_self_prohibited(admin_id)
            await uow.users.set_active(user_id, is_active=is_active)
            return ToggleActiveResult(id=user_id, is_active=is_active)

    async def change_role(self, command: ChangeRoleCommand, admin_id: int) -> UserDTO:
        """Change a user's role. If downgrading from CURATOR, clear all curator assignments."""
        del admin_id
        async with self._uow_factory.create(master=True) as uow:
            user = await uow.users.get_by_id(command.user_id)

            if user is None:
                msg = "User"
                raise EntityNotFoundError(msg, command.user_id)

            old_role = user.role
            new_role = UserRole.validate(command.new_role)
            user.change_role(new_role)
            await uow.users.update_role(command.user_id, new_role)

            # Clear assignments when downgrading from CURATOR.
            if old_role == UserRole.CURATOR and new_role != UserRole.CURATOR:
                await uow.assignments.clear_all_for_manager(command.user_id)

            return UserDTO(
                id=user.id if user.id is not None else 0,
                email=user.email,
                role=user.role,
                kind=user.kind,
                is_active=user.is_active,
            )
