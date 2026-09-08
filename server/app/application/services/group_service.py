"""Application service for group management with role-based access control."""

from __future__ import annotations

from domain.exceptions import EntityNotFound, ValidationError
from domain.value_objects.roles import UserKind
from domain.value_objects.user_context import UserContext

from application.ports.unit_of_work_factory import UnitOfWorkFactory


class GroupService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list_for_user(self, user_id: int, user_role: str, user_kind: str):
        async with self._uow_factory.create() as uow:
            ctx = UserContext(
                user_id=user_id,
                user_kind=user_kind,
                user_role=user_role,
                group_ids=await uow.groups.get_user_group_ids(user_id)
                if user_kind == UserKind.INTERNAL
                else [],
            )
            if ctx.is_admin:
                return await uow.groups.list_all()
            elif ctx.is_client:
                return []
            else:
                return await uow.groups.list_by_ids(ctx.group_ids) if ctx.group_ids else []

    async def create(self, name: str):
        async with self._uow_factory.create(master=True) as uow:
            group_id = await uow.groups.create(name)
            return group_id

    async def list_members(self, group_id: int):
        async with self._uow_factory.create() as uow:
            return await uow.groups.list_members(group_id)

    async def add_member(self, group_id: int, user_id: int):
        async with self._uow_factory.create(master=True) as uow:
            target = await uow.users.get_by_id(user_id)
            if target is None:
                raise EntityNotFound("User", user_id)
            if target.kind != UserKind.INTERNAL.value:
                raise ValidationError(
                    "Only internal employees can be added to groups",
                    field="user_id",
                )
            await uow.groups.add_user(user_id, group_id)

    async def remove_member(self, group_id: int, user_id: int):
        async with self._uow_factory.create(master=True) as uow:
            await uow.groups.remove_user(user_id, group_id)
