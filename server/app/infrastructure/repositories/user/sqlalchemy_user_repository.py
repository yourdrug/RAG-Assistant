"""SQLAlchemy ORM implementation of UserRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, func, literal, select
from sqlalchemy.dialects.postgresql import insert

from domain.identity.roles import UserKind, UserRole
from domain.identity.user import User
from infrastructure.database.models import ClientSpaceModel, UserModel


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


class SQLAlchemyUserRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, user_id: int) -> User | None:
        result = await self._db.execute(select(UserModel).where(UserModel.id == user_id))
        orm = result.scalar_one_or_none()
        return (await self.to_entities([orm]))[0] if orm else None

    async def get_by_ids(self, user_ids: list[int]) -> list[User]:
        if not user_ids:
            return []

        result = await self._db.execute(select(UserModel).where(UserModel.id.in_(user_ids)))
        return await self.to_entities(result.scalars().all())

    async def get_by_email(self, email: str) -> User | None:
        result = await self._db.execute(select(UserModel).where(UserModel.email == email.lower()))
        orm = result.scalar_one_or_none()
        return (await self.to_entities([orm]))[0] if orm else None

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

        if user.client_space_ids:
            await self.assign_client_spaces(orm.id, user.client_space_ids)

        return (await self.to_entities([orm]))[0]

    async def ensure_admin(self, email: str, hashed_password: str, role: str, kind: str) -> None:
        stmt = (
            insert(UserModel)
            .values(email=email.lower(), hashed_password=hashed_password, role=role, kind=kind)
            .on_conflict_do_nothing(index_elements=["email"])
        )
        await self._db.execute(stmt)
        await self._db.flush()

    async def exists_admin(self) -> bool:
        result = await self._db.execute(select(UserModel.id).where(UserModel.role == UserRole.ADMIN).limit(1))
        return result.scalar_one_or_none() is not None

    async def list_all(self, limit: int | None = None, offset: int = 0) -> list[User]:
        stmt = select(UserModel).order_by(UserModel.creation_date).offset(offset)

        if limit is not None:
            stmt = stmt.limit(limit)

        result = await self._db.execute(stmt)
        return await self.to_entities(result.scalars().all())

    async def count_all(self) -> int:
        result = await self._db.execute(select(func.count()).select_from(UserModel))
        return int(result.scalar_one())

    async def set_active(self, user_id: int, *, is_active: bool) -> bool:
        result = await self._db.execute(select(UserModel).where(UserModel.id == user_id))
        orm = result.scalar_one_or_none()

        if orm is None:
            return False

        orm.is_active = is_active
        await self._db.flush()
        return True

    async def update_role(self, user_id: int, role: str) -> bool:
        result = await self._db.execute(select(UserModel).where(UserModel.id == user_id))
        orm = result.scalar_one_or_none()

        if orm is None:
            return False

        orm.role = role
        await self._db.flush()
        return True

    async def assign_client_spaces(self, user_id: int, space_ids: frozenset[int]) -> None:
        await self._db.execute(delete(ClientSpaceModel).where(ClientSpaceModel.user_id == user_id))

        for space_id in sorted(space_ids):
            # Rows only for a CLIENT user: the database has no cross-table check for the kind.
            await self._db.execute(
                insert(ClientSpaceModel).from_select(
                    ["user_id", "space_id"],
                    select(UserModel.id, literal(space_id)).where(
                        UserModel.id == user_id, UserModel.kind == UserKind.CLIENT
                    ),
                )
            )

        await self._db.flush()

    async def to_entities(self, orms: Sequence[UserModel]) -> list[User]:
        client_ids = [orm.id for orm in orms if orm.kind == UserKind.CLIENT]
        spaces: dict[int, set[int]] = {}

        if client_ids:
            rows = await self._db.execute(
                select(ClientSpaceModel.user_id, ClientSpaceModel.space_id).where(
                    ClientSpaceModel.user_id.in_(client_ids)
                )
            )

            for user_id, space_id in rows.all():
                spaces.setdefault(user_id, set()).add(space_id)

        return [self.to_entity(orm, frozenset(spaces.get(orm.id, ()))) for orm in orms]

    @staticmethod
    def to_entity(orm: UserModel, client_space_ids: frozenset[int]) -> User:
        return User(
            id=orm.id,
            email=orm.email,
            hashed_password=orm.hashed_password,
            role=UserRole(orm.role),
            kind=UserKind(orm.kind),
            is_active=orm.is_active,
            client_space_ids=client_space_ids,
            creation_date=orm.creation_date,
        )
