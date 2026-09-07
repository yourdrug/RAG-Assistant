"""SQLAlchemy implementation of ConfigParameterRepository."""

from __future__ import annotations

from domain.repositories.config_parameter_repository import ConfigParameter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ConfigParameterModel


class SQLAlchemyConfigParameterRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_all(self) -> list[ConfigParameter]:
        result = await self._db.execute(
            select(ConfigParameterModel).order_by(ConfigParameterModel.category, ConfigParameterModel.key)
        )
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def get_by_key(self, key: str) -> ConfigParameter | None:
        result = await self._db.execute(select(ConfigParameterModel).where(ConfigParameterModel.key == key))
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def get_by_key_and_domain(
        self, key: str, domain_key: str | None = None,
    ) -> ConfigParameter | None:
        stmt = select(ConfigParameterModel).where(ConfigParameterModel.key == key)
        if domain_key is not None:
            stmt = stmt.where(ConfigParameterModel.domain_key == domain_key)
        else:
            stmt = stmt.where(ConfigParameterModel.domain_key.is_(None))
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def update_value(self, key: str, value: str, domain_key: str | None = None) -> None:
        stmt = select(ConfigParameterModel).where(ConfigParameterModel.key == key).with_for_update()
        if domain_key is not None:
            stmt = stmt.where(ConfigParameterModel.domain_key == domain_key)
        else:
            stmt = stmt.where(ConfigParameterModel.domain_key.is_(None))
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm:
            orm.value = value
            await self._db.flush()

    async def update_category(self, key: str, category: str) -> None:
        result = await self._db.execute(
            select(ConfigParameterModel).where(ConfigParameterModel.key == key).with_for_update()
        )
        orm = result.scalar_one_or_none()
        if orm and orm.category != category:
            orm.category = category
            await self._db.flush()

    async def save(self, entity: ConfigParameter) -> None:
        orm = ConfigParameterModel(
            key=entity.key,
            value=entity.value,
            value_type=entity.value_type,
            category=entity.category,
            description=entity.description,
            min_value=entity.min_value,
            max_value=entity.max_value,
            allowed_values={"values": entity.allowed_values} if entity.allowed_values else None,
            domain_key=entity.domain_key,
        )
        self._db.add(orm)
        await self._db.flush()

    async def upsert(self, entity: ConfigParameter) -> None:
        existing = await self.get_by_key_and_domain(entity.key, entity.domain_key)
        if existing:
            await self.update_value(entity.key, entity.value, entity.domain_key)
        else:
            await self.save(entity)

    async def count(self) -> int:
        result = await self._db.execute(select(func.count()).select_from(ConfigParameterModel))
        return result.scalar_one()

    @staticmethod
    def _to_entity(orm: ConfigParameterModel) -> ConfigParameter:
        allowed = orm.allowed_values
        if isinstance(allowed, dict):
            allowed = allowed.get("values")
        return ConfigParameter(
            key=orm.key,
            value=orm.value,
            value_type=orm.value_type,
            category=orm.category,
            description=orm.description,
            min_value=orm.min_value,
            max_value=orm.max_value,
            allowed_values=allowed,
            domain_key=orm.domain_key,
        )
