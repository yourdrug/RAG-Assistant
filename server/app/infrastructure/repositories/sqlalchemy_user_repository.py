"""SQLAlchemy ORM implementation of UserRepository."""

from __future__ import annotations

from domain.entities.user import User
from domain.value_objects.roles import UserKind, UserRole
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import UserModel


class SQLAlchemyUserRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, user_id: int) -> User | None:
        result = await self._db.execute(select(UserModel).where(UserModel.id == user_id))
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def get_by_email(self, email: str) -> User | None:
        result = await self._db.execute(select(UserModel).where(UserModel.email == email.lower()))
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def save(self, user: User) -> User:
        orm = UserModel(
            email=user.email.lower(),
            hashed_password=user.hashed_password,
            role=user.role,
            kind=user.kind,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        return self._to_entity(orm)

    async def exists_admin(self) -> bool:
        result = await self._db.execute(select(UserModel.id).where(UserModel.role == "admin").limit(1))
        return result.scalar_one_or_none() is not None

    async def list_all(self) -> list[User]:
        result = await self._db.execute(select(UserModel).order_by(UserModel.creation_date))
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def set_active(self, user_id: int, is_active: bool) -> bool:
        result = await self._db.execute(select(UserModel).where(UserModel.id == user_id))
        orm = result.scalar_one_or_none()
        if orm is None:
            return False
        orm.is_active = is_active
        await self._db.flush()
        return True

    @staticmethod
    def _to_entity(orm: UserModel) -> User:
        return User(
            id=orm.id,
            email=orm.email,
            hashed_password=orm.hashed_password,
            role=UserRole(orm.role),
            kind=UserKind(orm.kind),
            is_active=orm.is_active,
            creation_date=orm.creation_date,
        )
