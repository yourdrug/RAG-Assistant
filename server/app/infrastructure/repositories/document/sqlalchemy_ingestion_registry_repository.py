"""SQLAlchemy implementation of IngestionRegistryRepository."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from domain.repositories.ingestion_registry_repository import (
    IngestionRegistryEntry,
)
from infrastructure.database.models import IngestionRegistryModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


log = logging.getLogger("default")


class SQLAlchemyIngestionRegistryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, filename: str) -> IngestionRegistryEntry | None:
        stmt = select(IngestionRegistryModel).where(IngestionRegistryModel.ingestion_key == filename)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        return self._to_entry(orm)

    async def upsert(self, entry: IngestionRegistryEntry) -> None:
        stmt = select(IngestionRegistryModel).where(
            IngestionRegistryModel.ingestion_key == entry.ingestion_key
        )
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is not None:
            orm.file_hash = entry.file_hash
            orm.source = entry.source
            orm.chunks = entry.chunks
            orm.chars = entry.chars
            orm.indexed_at = entry.indexed_at
        else:
            orm = IngestionRegistryModel(
                filename=entry.filename,
                ingestion_key=entry.ingestion_key,
                file_hash=entry.file_hash,
                source=entry.source,
                chunks=entry.chunks,
                chars=entry.chars,
                indexed_at=entry.indexed_at,
            )
            self._session.add(orm)
        await self._session.flush()

    async def delete(self, filename: str) -> None:
        stmt = select(IngestionRegistryModel).where(IngestionRegistryModel.ingestion_key == filename)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is not None:
            await self._session.delete(orm)
            await self._session.flush()

    async def list_all(self) -> dict[str, IngestionRegistryEntry]:
        stmt = select(IngestionRegistryModel).order_by(IngestionRegistryModel.filename)
        result = await self._session.execute(stmt)
        orms = result.scalars().all()
        return {orm.ingestion_key: self._to_entry(orm) for orm in orms}

    async def is_already_indexed(self, filename: str, file_hash: str) -> bool:
        entry = await self.get(filename)
        if entry is None:
            return False
        return entry.file_hash == file_hash

    @staticmethod
    def _to_entry(orm: IngestionRegistryModel) -> IngestionRegistryEntry:
        return IngestionRegistryEntry(
            filename=orm.filename,
            ingestion_key=orm.ingestion_key,
            file_hash=orm.file_hash,
            source=orm.source,
            chunks=orm.chunks,
            chars=orm.chars,
            indexed_at=orm.indexed_at,
        )
