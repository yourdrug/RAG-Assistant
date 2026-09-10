"""SQLAlchemy implementation of AssignmentRepository."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from domain.repositories.assignment_repository import AssignmentRepository
from domain.value_objects.roles import UserKind
from infrastructure.database.models import (
    CuratorGroupAssignmentModel,
    CuratorUserAssignmentModel,
    UserModel,
)


class SQLAlchemyAssignmentRepository(AssignmentRepository):
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_managed_user_ids(self, manager_id: int) -> list[int]:
        stmt = select(CuratorUserAssignmentModel.target_user_id).where(
            CuratorUserAssignmentModel.curator_id == manager_id
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_managed_client_ids(self, manager_id: int) -> list[int]:
        stmt = (
            select(CuratorUserAssignmentModel.target_user_id)
            .join(UserModel, UserModel.id == CuratorUserAssignmentModel.target_user_id)
            .where(
                CuratorUserAssignmentModel.curator_id == manager_id,
                UserModel.kind == UserKind.CLIENT,
            )
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_managed_internal_ids(self, manager_id: int) -> list[int]:
        stmt = (
            select(CuratorUserAssignmentModel.target_user_id)
            .join(UserModel, UserModel.id == CuratorUserAssignmentModel.target_user_id)
            .where(
                CuratorUserAssignmentModel.curator_id == manager_id,
                UserModel.kind == UserKind.INTERNAL,
            )
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_managed_group_ids(self, manager_id: int) -> list[int]:
        stmt = select(CuratorGroupAssignmentModel.group_id).where(
            CuratorGroupAssignmentModel.curator_id == manager_id
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def assign_user(self, manager_id: int, target_user_id: int, assigned_by: int) -> None:
        entry = CuratorUserAssignmentModel(
            curator_id=manager_id,
            target_user_id=target_user_id,
            assigned_by=assigned_by,
        )
        self._db.add(entry)
        await self._db.flush()

    async def unassign_user(self, manager_id: int, target_user_id: int) -> None:
        stmt = delete(CuratorUserAssignmentModel).where(
            CuratorUserAssignmentModel.curator_id == manager_id,
            CuratorUserAssignmentModel.target_user_id == target_user_id,
        )
        await self._db.execute(stmt)
        await self._db.flush()

    async def assign_group(self, manager_id: int, group_id: int, assigned_by: int) -> None:
        entry = CuratorGroupAssignmentModel(
            curator_id=manager_id,
            group_id=group_id,
            assigned_by=assigned_by,
        )
        self._db.add(entry)
        await self._db.flush()

    async def unassign_group(self, manager_id: int, group_id: int) -> None:
        stmt = delete(CuratorGroupAssignmentModel).where(
            CuratorGroupAssignmentModel.curator_id == manager_id,
            CuratorGroupAssignmentModel.group_id == group_id,
        )
        await self._db.execute(stmt)
        await self._db.flush()

    async def clear_all_for_manager(self, manager_id: int) -> None:
        await self._db.execute(
            delete(CuratorUserAssignmentModel).where(CuratorUserAssignmentModel.curator_id == manager_id)
        )
        await self._db.execute(
            delete(CuratorGroupAssignmentModel).where(CuratorGroupAssignmentModel.curator_id == manager_id)
        )
        await self._db.flush()

    async def list_managers_for_group(self, group_id: int) -> list[int]:
        stmt = select(CuratorGroupAssignmentModel.curator_id).where(
            CuratorGroupAssignmentModel.group_id == group_id
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())
