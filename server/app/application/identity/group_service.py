"""Application service for group management with role-based access control."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.identity.user_context_factory import UserContextFactory
from domain.identity.roles import UserKind
from domain.shared.exceptions import EntityNotFound, ValidationError


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory
    from domain.identity.group import GroupInfo, GroupMemberInfo


class GroupService:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, user_ctx_factory: UserContextFactory | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def list_for_user(self, user_id: int, user_role: str, user_kind: str) -> list[GroupInfo]:
        async with self._uow_factory.create() as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)

            if ctx.is_admin:
                return await uow.groups.list_all()

            if ctx.is_client:
                return []

            if ctx.is_curator:
                # CURATOR sees own membership groups + managed groups
                all_group_ids = list(set(ctx.group_ids) | set(ctx.managed_group_ids))
                return await uow.groups.list_by_ids(all_group_ids) if all_group_ids else []

            return await uow.groups.list_by_ids(list(ctx.group_ids)) if ctx.group_ids else []

    async def create(self, name: str) -> int:
        async with self._uow_factory.create(master=True) as uow:
            return await uow.groups.create(name)

    async def list_members(self, group_id: int) -> list[GroupMemberInfo]:
        async with self._uow_factory.create() as uow:
            return await uow.groups.list_members(group_id)

    async def add_member(self, group_id: int, user_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            target = await uow.users.get_by_id(user_id)

            if target is None:
                entity = "User"
                raise EntityNotFound(entity, user_id)

            if target.kind != UserKind.INTERNAL.value:
                msg = "Only internal employees can be added to groups"
                raise ValidationError(
                    msg,
                    field="user_id",
                )

            await uow.groups.add_user(user_id, group_id)

    async def remove_member(self, group_id: int, user_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.groups.remove_user(user_id, group_id)
