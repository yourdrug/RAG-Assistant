"""SQLAlchemy implementation of ActVersionRepository."""

from __future__ import annotations

from datetime import date, datetime
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.act_version import ActVersion
from infrastructure.database.models import ActVersionModel, DocumentModel


def _as_date(value) -> date | None:
    """Normalize a Date/DateTime column value into a date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    return value


class SQLAlchemyActVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _to_entity(m: ActVersionModel) -> ActVersion:
        return ActVersion(
            id=m.id,
            act_id=m.act_id,
            document_id=m.document_id,
            effective_from=_as_date(m.effective_from),
            effective_to=_as_date(m.effective_to),
            is_current=m.is_current,
            date_source=m.date_source,
            date_confidence=m.date_confidence,
            verified_by=m.verified_by,
            verified_at=m.verified_at,
        )

    async def create(self, version: ActVersion) -> ActVersion:
        m = ActVersionModel(
            act_id=version.act_id,
            document_id=version.document_id,
            effective_from=version.effective_from,
            effective_to=version.effective_to,
            is_current=version.is_current,
            date_source=version.date_source,
            date_confidence=version.date_confidence,
            verified_by=version.verified_by,
            verified_at=version.verified_at,
        )
        self._session.add(m)
        await self._session.flush()
        version.id = m.id
        return version

    async def get_by_id(self, version_id: int) -> ActVersion | None:
        m = await self._session.get(ActVersionModel, version_id)
        return self._to_entity(m) if m else None

    async def get_by_document_id(self, document_id: int, *, for_update: bool = False) -> ActVersion | None:
        if for_update:
            # Lock the document even before its first version exists, so two
            # workers reindexing it cannot both create a new version.
            await self._session.execute(
                select(DocumentModel.id).where(DocumentModel.id == document_id).with_for_update()
            )
        stmt = (
            select(ActVersionModel)
            .where(ActVersionModel.document_id == document_id)
            .order_by(
                (ActVersionModel.date_source == "manual").desc(),
                ActVersionModel.verified_at.desc().nullslast(),
                ActVersionModel.id.desc(),
            )
            .limit(1)
        )
        result = await self._session.execute(stmt)
        m = result.scalar_one_or_none()
        return self._to_entity(m) if m else None

    async def unset_current(self, act_id: int) -> None:
        stmt = (
            update(ActVersionModel)
            .where(ActVersionModel.act_id == act_id, ActVersionModel.is_current.is_(True))
            .values(is_current=False)
        )
        await self._session.execute(stmt)

    async def list_by_act(self, act_id: int) -> list[ActVersion]:
        stmt = select(ActVersionModel).where(ActVersionModel.act_id == act_id).order_by(ActVersionModel.id)
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars()]

    async def list_pending_review(self) -> list[ActVersion]:
        stmt = (
            select(ActVersionModel)
            .where((ActVersionModel.date_source == "extracted") | (ActVersionModel.act_id.is_(None)))
            .order_by(ActVersionModel.id)
        )
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars()]

    async def list_page(self, limit: int, offset: int) -> tuple[list[ActVersion], int]:
        total = await self._session.scalar(select(func.count()).select_from(ActVersionModel))
        stmt = select(ActVersionModel).order_by(ActVersionModel.id).offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars()], int(total or 0)

    async def update(self, version: ActVersion) -> None:
        m = await self._session.get(ActVersionModel, version.id)
        if m is None:
            return
        m.effective_from = version.effective_from
        m.effective_to = version.effective_to
        m.act_id = version.act_id
        m.is_current = version.is_current
        m.date_source = version.date_source
        m.date_confidence = version.date_confidence
        m.verified_by = version.verified_by
        m.verified_at = version.verified_at
        await self._session.flush()
