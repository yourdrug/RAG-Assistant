"""RegulatoryAct repository interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.regulatory_act import RegulatoryAct


@runtime_checkable
class RegulatoryActRepository(Protocol):
    async def find_by_type_and_number(self, act_type: str, act_number: str) -> RegulatoryAct | None: ...
    async def get_by_id(self, act_id: int) -> RegulatoryAct | None: ...
    async def save(self, act: RegulatoryAct) -> RegulatoryAct: ...
    async def list_all(self) -> list[RegulatoryAct]: ...
