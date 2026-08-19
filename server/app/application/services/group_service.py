"""Application service for group management with role-based access control."""

from __future__ import annotations

from domain.exceptions import EntityNotFound, ValidationError
from domain.value_objects.roles import UserKind, UserRole

from application.ports.unit_of_work_factory import UnitOfWorkFactory


class GroupService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list_for_user(self, user_id: int, user_role: str, user_kind: str):
        async with self._uow_factory.create() as uow:
            if user_role == UserRole.ADMIN.value:
                return await uow.groups.list_all()
            elif user_kind != UserKind.INTERNAL.value:
                return []
            else:
                group_ids = await uow.groups.get_user_group_ids(user_id)
                return await uow.groups.list_by_ids(group_ids) if group_ids else []

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
                    detail="Only internal employees can be added to groups",
                    field="user_id",
                )
            await uow.groups.add_user(user_id, group_id)

    async def remove_member(self, group_id: int, user_id: int):
        async with self._uow_factory.create(master=True) as uow:
            await uow.groups.remove_user(user_id, group_id)
