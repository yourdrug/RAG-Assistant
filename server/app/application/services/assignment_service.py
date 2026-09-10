"""Application service for assignment management (user/group assignment CRUD + validation)."""

from __future__ import annotations

import logging

from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.exceptions import BusinessRuleViolation, EntityNotFound
from domain.value_objects.roles import UserRole

log = logging.getLogger("default")


class AssignmentService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def get_scope(self, manager_id: int) -> dict:
        """Return managed_client_ids, managed_internal_ids, managed_group_ids for a manager."""
        async with self._uow_factory.create() as uow:
            assignments = uow.assignments
            return {
                "managed_client_ids": await assignments.get_managed_client_ids(manager_id),
                "managed_internal_ids": await assignments.get_managed_internal_ids(manager_id),
                "managed_group_ids": await assignments.get_managed_group_ids(manager_id),
            }

    async def assign_user(self, manager_id: int, target_user_id: int, assigned_by: int) -> None:
        """Assign a user to a manager. Validates both users exist and have correct roles."""
        async with self._uow_factory.create(master=True) as uow:
            manager = await uow.users.get_by_id(manager_id)
            if manager is None:
                raise EntityNotFound("User", manager_id)
            if manager.role != UserRole.CURATOR:
                raise BusinessRuleViolation("Target user is not a curator")

            target = await uow.users.get_by_id(target_user_id)
            if target is None:
                raise EntityNotFound("User", target_user_id)
            if target.role in (UserRole.ADMIN, UserRole.CURATOR):
                raise BusinessRuleViolation("Cannot assign admin or curator users to a curator")
            if manager_id == target_user_id:
                raise BusinessRuleViolation("Cannot assign a curator to themselves")

            await uow.assignments.assign_user(manager_id, target_user_id, assigned_by)
            log.info(
                "User %d assigned user %d (by %d)",
                manager_id,
                target_user_id,
                assigned_by,
            )

    async def unassign_user(self, manager_id: int, target_user_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.assignments.unassign_user(manager_id, target_user_id)
            log.info("User %d unassigned user %d", manager_id, target_user_id)

    async def assign_group(self, manager_id: int, group_id: int, assigned_by: int) -> None:
        """Assign a group to a manager. Validates manager exists and has correct role."""
        async with self._uow_factory.create(master=True) as uow:
            manager = await uow.users.get_by_id(manager_id)
            if manager is None:
                raise EntityNotFound("User", manager_id)
            if manager.role != UserRole.CURATOR:
                raise BusinessRuleViolation("Target user is not a curator")

            groups = await uow.groups.list_by_ids([group_id])
            if not groups:
                raise EntityNotFound("Group", group_id)

            await uow.assignments.assign_group(manager_id, group_id, assigned_by)
            log.info(
                "User %d assigned group %d (by %d)",
                manager_id,
                group_id,
                assigned_by,
            )

    async def unassign_group(self, manager_id: int, group_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.assignments.unassign_group(manager_id, group_id)
            log.info("User %d unassigned group %d", manager_id, group_id)
