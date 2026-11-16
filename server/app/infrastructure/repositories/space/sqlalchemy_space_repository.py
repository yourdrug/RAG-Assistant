"""SQLAlchemy repository for knowledge spaces and the groups admitted to fill them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, exists, select
from sqlalchemy.exc import IntegrityError

from domain.knowledge.space import KnowledgeSpace
from domain.shared.exceptions import UniqueConstraintError
from infrastructure.database.models import (
    DocumentModel,
    KnowledgeSpaceModel,
    KnowledgeSpaceUploadGroupModel,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


class SQLAlchemyKnowledgeSpaceRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, space_id: int, *, for_update: bool = False) -> KnowledgeSpace | None:
        stmt = select(KnowledgeSpaceModel).where(KnowledgeSpaceModel.id == space_id)

        if for_update:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)

        orm = (await self._db.execute(stmt)).scalar_one_or_none()

        if orm is None:
            return None

        groups = await self.upload_groups([orm.id])
        return self.to_entity(orm, groups.get(orm.id, ()))

    async def list_all(self) -> list[KnowledgeSpace]:
        rows = (
            await self._db.execute(select(KnowledgeSpaceModel).order_by(KnowledgeSpaceModel.id))
        ).scalars()
        spaces = list(rows)
        groups = await self.upload_groups([orm.id for orm in spaces])
        return [self.to_entity(orm, groups.get(orm.id, ())) for orm in spaces]

    async def create(self, space: KnowledgeSpace) -> KnowledgeSpace:
        orm = KnowledgeSpaceModel(name=space.name, pack_key=space.pack_key, creation_date=space.creation_date)
        self._db.add(orm)
        await self.flush()
        space.id = orm.id
        await self.replace_upload_groups(orm.id, space.upload_group_ids)
        return space

    async def update(self, space: KnowledgeSpace) -> KnowledgeSpace:
        if space.id is None:
            msg = "Cannot update a space without id"
            raise ValueError(msg)

        orm = await self._db.get(KnowledgeSpaceModel, space.id)

        if orm is None:
            msg = f"Space {space.id} does not exist"
            raise ValueError(msg)

        orm.name = space.name
        await self.flush()
        await self.replace_upload_groups(space.id, space.upload_group_ids)
        return space

    async def delete(self, space_id: int) -> None:
        await self._db.execute(delete(KnowledgeSpaceModel).where(KnowledgeSpaceModel.id == space_id))

    async def has_documents(self, space_id: int) -> bool:
        stmt = select(exists().where(DocumentModel.space_id == space_id))
        return bool((await self._db.execute(stmt)).scalar())

    async def replace_upload_groups(self, space_id: int, group_ids: frozenset[int]) -> None:
        await self._db.execute(
            delete(KnowledgeSpaceUploadGroupModel).where(KnowledgeSpaceUploadGroupModel.space_id == space_id)
        )
        self._db.add_all(
            KnowledgeSpaceUploadGroupModel(space_id=space_id, group_id=group_id)
            for group_id in sorted(group_ids)
        )
        await self.flush()

    async def upload_groups(self, space_ids: Sequence[int]) -> dict[int, list[int]]:
        if not space_ids:
            return {}

        rows = await self._db.execute(
            select(KnowledgeSpaceUploadGroupModel.space_id, KnowledgeSpaceUploadGroupModel.group_id).where(
                KnowledgeSpaceUploadGroupModel.space_id.in_(space_ids)
            )
        )
        groups: dict[int, list[int]] = {}

        for space_id, group_id in rows.all():
            groups.setdefault(space_id, []).append(group_id)

        return groups

    async def flush(self) -> None:
        try:
            await self._db.flush()
        except IntegrityError as exc:
            raise UniqueConstraintError(detail=str(exc)) from exc

    @staticmethod
    def to_entity(orm: KnowledgeSpaceModel, group_ids: Sequence[int]) -> KnowledgeSpace:
        return KnowledgeSpace(
            id=orm.id,
            name=orm.name,
            pack_key=orm.pack_key,
            upload_group_ids=frozenset(group_ids),
            creation_date=orm.creation_date,
        )
