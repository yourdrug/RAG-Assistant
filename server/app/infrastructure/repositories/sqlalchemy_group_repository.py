"""SQLAlchemy ORM implementation of GroupRepository."""

from __future__ import annotations

from domain.value_objects.query_results import GroupInfo, GroupMemberInfo
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import GroupModel, UserGroupModel, UserModel


class SQLAlchemyGroupRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, name: str) -> int:
        orm = GroupModel(name=name)
        self._db.add(orm)
        await self._db.flush()
        return orm.id

    async def list_all(self) -> list[GroupInfo]:
        result = await self._db.execute(select(GroupModel).order_by(GroupModel.name))
        return [GroupInfo(id=orm.id, name=orm.name) for orm in result.scalars().all()]

    async def list_by_ids(self, ids: list[int]) -> list[GroupInfo]:
        if not ids:
            return []
        result = await self._db.execute(
            select(GroupModel).where(GroupModel.id.in_(ids)).order_by(GroupModel.name)
        )
        return [GroupInfo(id=orm.id, name=orm.name) for orm in result.scalars().all()]

    async def get_user_group_ids(self, user_id: int) -> list[int]:
        result = await self._db.execute(
            select(UserGroupModel.group_id).where(UserGroupModel.user_id == user_id)
        )
        return [row[0] for row in result.all()]

    async def add_user(self, user_id: int, group_id: int) -> None:
        orm = UserGroupModel(user_id=user_id, group_id=group_id)
        self._db.add(orm)
        await self._db.flush()

    async def remove_user(self, user_id: int, group_id: int) -> None:
        result = await self._db.execute(
            select(UserGroupModel).where(
                UserGroupModel.user_id == user_id,
                UserGroupModel.group_id == group_id,
            )
        )
        orm = result.scalar_one_or_none()
        if orm:
            await self._db.delete(orm)
            await self._db.flush()

    async def list_members(self, group_id: int) -> list[GroupMemberInfo]:
        result = await self._db.execute(
            select(UserModel.id, UserModel.email)
            .join(UserGroupModel, UserGroupModel.user_id == UserModel.id)
            .where(UserGroupModel.group_id == group_id)
            .order_by(UserModel.email)
        )
        return [GroupMemberInfo(id=row.id, email=row.email) for row in result.all()]
