"""SQLAlchemy implementation of IngestionRegistryRepository."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from application.ports.session_protocol import SessionProtocol
from domain.repositories.ingestion_registry_repository import (
    IngestionRegistryEntry,
)
from infrastructure.database.models import IngestionRegistryModel

log = logging.getLogger("default")


class SQLAlchemyIngestionRegistryRepository:
    def __init__(self, session: SessionProtocol) -> None:
        self._session = session

    async def get(self, object_key: str) -> IngestionRegistryEntry | None:
        stmt = select(IngestionRegistryModel).where(IngestionRegistryModel.object_key == object_key)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        return self.to_entry(orm)

    async def upsert(self, entry: IngestionRegistryEntry) -> None:
        values = {
            "object_key": entry.object_key,
            "filename": entry.filename,
            "file_hash": entry.file_hash,
            "source": entry.source,
            "chunks": entry.chunks,
            "chars": entry.chars,
            "indexed_at": entry.indexed_at,
        }
        stmt = insert(IngestionRegistryModel).values(**values)
        await self._session.execute(
            stmt.on_conflict_do_update(
                index_elements=[IngestionRegistryModel.object_key],
                set_={name: getattr(stmt.excluded, name) for name in values if name != "object_key"},
            )
        )
        await self._session.flush()

    async def delete(self, object_key: str) -> None:
        stmt = select(IngestionRegistryModel).where(IngestionRegistryModel.object_key == object_key)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is not None:
            await self._session.delete(orm)
            await self._session.flush()

    async def list_all(self) -> dict[str, IngestionRegistryEntry]:
        stmt = select(IngestionRegistryModel).order_by(IngestionRegistryModel.object_key)
        result = await self._session.execute(stmt)
        orms = result.scalars().all()
        return {orm.object_key: self.to_entry(orm) for orm in orms}

    async def is_already_indexed(self, object_key: str, file_hash: str) -> bool:
        entry = await self.get(object_key)
        if entry is None:
            return False
        return entry.file_hash == file_hash

    @staticmethod
    def to_entry(orm: IngestionRegistryModel) -> IngestionRegistryEntry:
        return IngestionRegistryEntry(
            object_key=orm.object_key,
            filename=orm.filename,
            file_hash=orm.file_hash,
            source=orm.source,
            chunks=orm.chunks,
            chars=orm.chars,
            indexed_at=orm.indexed_at,
        )
