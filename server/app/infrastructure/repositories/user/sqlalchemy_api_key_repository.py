"""SQLAlchemy ORM implementation of ApiKeyRepository."""

from __future__ import annotations

from datetime import UTC, datetime

from domain.entities.api_key import ApiKey
from domain.value_objects.query_results import ApiKeyClientInfo
from domain.value_objects.roles import UserKind
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ApiKeyModel, UserModel


class SQLAlchemyApiKeyRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, user_id: int, key_hash: str, key_prefix: str, name: str | None = None) -> ApiKey:
        orm = ApiKeyModel(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        return self._to_entity(orm)

    async def list_for_user(self, user_id: int) -> list[ApiKey]:
        result = await self._db.execute(
            select(ApiKeyModel)
            .where(ApiKeyModel.user_id == user_id)
            .order_by(ApiKeyModel.creation_date.desc())
        )
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def revoke(self, api_key_id: int, user_id: int | None = None) -> bool:
        stmt = select(ApiKeyModel).where(
            ApiKeyModel.id == api_key_id,
            ApiKeyModel.revoked_at.is_(None),
        )
        if user_id is not None:
            stmt = stmt.where(ApiKeyModel.user_id == user_id)

        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            return False

        orm.revoked_at = datetime.now(tz=UTC)
        await self._db.flush()
        return True

    async def get_active_client_by_hash(self, key_hash: str) -> ApiKeyClientInfo | None:
        result = await self._db.execute(
            select(ApiKeyModel, UserModel)
            .join(UserModel, UserModel.id == ApiKeyModel.user_id)
            .where(
                ApiKeyModel.key_hash == key_hash,
                ApiKeyModel.revoked_at.is_(None),
                UserModel.kind == UserKind.CLIENT,
                UserModel.is_active.is_(True),
            )
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None

        api_key_orm, user_orm = row
        return ApiKeyClientInfo(
            api_key_id=api_key_orm.id,
            id=user_orm.id,
            email=user_orm.email,
            role=user_orm.role,
            kind=user_orm.kind,
            is_active=user_orm.is_active,
        )

    async def touch_last_used(self, api_key_id: int) -> None:
        result = await self._db.execute(select(ApiKeyModel).where(ApiKeyModel.id == api_key_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.last_used_at = datetime.now(tz=UTC)
            await self._db.flush()

    @staticmethod
    def _to_entity(orm: ApiKeyModel) -> ApiKey:
        return ApiKey(
            id=orm.id,
            user_id=orm.user_id,
            key_hash=orm.key_hash,
            key_prefix=orm.key_prefix,
            name=orm.name,
            creation_date=orm.creation_date,
            revoked_at=orm.revoked_at,
            last_used_at=orm.last_used_at,
        )
