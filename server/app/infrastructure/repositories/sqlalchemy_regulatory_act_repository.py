"""SQLAlchemy implementation of RegulatoryActRepository."""

from __future__ import annotations

from sqlalchemy import select
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
        )

    async def find_by_type_and_number(self, act_type: str, act_number: str) -> RegulatoryAct | None:
        stmt = select(RegulatoryActModel).where(
            RegulatoryActModel.act_type == act_type,
            RegulatoryActModel.act_number == act_number,
        )
        result = await self._session.execute(stmt)
        m = result.scalar_one_or_none()
        return self._to_entity(m) if m else None

    async def get_by_id(self, act_id: int) -> RegulatoryAct | None:
        m = await self._session.get(RegulatoryActModel, act_id)
        return self._to_entity(m) if m else None

    async def save(self, act: RegulatoryAct) -> RegulatoryAct:
        m = RegulatoryActModel(
            act_type=act.act_type,
            act_number=act.act_number,
            title=act.title,
            issuing_authority=act.issuing_authority,
        )
        self._session.add(m)
        await self._session.flush()
        act.id = m.id
        return act

    async def list_all(self) -> list[RegulatoryAct]:
        stmt = select(RegulatoryActModel).order_by(RegulatoryActModel.id)
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars()]
