"""RegulatoryAct repository interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from datetime import date

from domain.entities.regulatory_act import RegulatoryAct


@runtime_checkable
class RegulatoryActRepository(Protocol):
    async def find_by_type_and_number(
        self,
        act_type: str,
        act_number: str,
        act_date: date | None = None,
        visibility_scope: str = "internal_public",
    ) -> RegulatoryAct | None: ...
    async def update_act_date(self, act_id: int, act_date: date) -> None: ...
    async def get_by_id(self, act_id: int) -> RegulatoryAct | None: ...
    async def save(self, act: RegulatoryAct) -> RegulatoryAct: ...
    async def list_all(self) -> list[RegulatoryAct]: ...
