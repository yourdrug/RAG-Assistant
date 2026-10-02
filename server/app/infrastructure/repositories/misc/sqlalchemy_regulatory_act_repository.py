"""SQLAlchemy implementation of RegulatoryActRepository."""

from __future__ import annotations

from datetime import date

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.regulatory_act import RegulatoryAct
from infrastructure.database.models import RegulatoryActModel


class SQLAlchemyRegulatoryActRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _to_entity(m: RegulatoryActModel) -> RegulatoryAct:
        return RegulatoryAct(
            id=m.id,
            act_type=m.act_type,
            act_number=m.act_number,
            title=m.title,
            issuing_authority=m.issuing_authority,
            act_date=m.act_date,
            visibility_scope=m.visibility_scope,
        )

    async def find_by_type_and_number(
        self,
        act_type: str,
        act_number: str | None,
        act_date: date | None = None,
        visibility_scope: str = "internal_public",
    ) -> RegulatoryAct | None:
        stmt = (
            select(RegulatoryActModel)
            .where(
                RegulatoryActModel.act_type == act_type,
                RegulatoryActModel.act_number == act_number,
                RegulatoryActModel.act_date == act_date,
                RegulatoryActModel.visibility_scope == visibility_scope,
            )
            .with_for_update()
        )
        result = await self._session.execute(stmt)
        m = result.scalar_one_or_none()
        return self._to_entity(m) if m else None

    async def update_act_date(self, act_id: int, act_date: date) -> None:
        await self._session.execute(
            update(RegulatoryActModel).where(RegulatoryActModel.id == act_id).values(act_date=act_date)
        )

    async def get_by_id(self, act_id: int) -> RegulatoryAct | None:
        m = await self._session.get(RegulatoryActModel, act_id)
        return self._to_entity(m) if m else None

    async def save(self, act: RegulatoryAct) -> RegulatoryAct:
        stmt = (
            insert(RegulatoryActModel)
            .values(
                act_type=act.act_type,
                act_number=act.act_number,
                title=act.title,
                issuing_authority=act.issuing_authority,
                act_date=act.act_date,
                visibility_scope=act.visibility_scope,
            )
            .on_conflict_do_nothing()
            .returning(RegulatoryActModel.id)
        )
        act.id = await self._session.scalar(stmt)
        if act.id is None:
            existing = await self.find_by_type_and_number(
                act.act_type, act.act_number, act.act_date, act.visibility_scope
            )
            if existing is None:
                raise RuntimeError("Regulatory act conflict did not resolve to an existing identity")
            return existing
        return act

    async def list_all(self) -> list[RegulatoryAct]:
        stmt = select(RegulatoryActModel).order_by(RegulatoryActModel.id)
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars()]
