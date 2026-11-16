"""Belarusian law pack test support: its profile registry and in-memory act repositories."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from domain.domain_packs.belarus_law.acts.repositories import ActRepositories
from domain.ingestion.profiles.registry import DomainProfileRegistry
from infrastructure.domain_packs.belarus_law.pack import BELARUS_LAW_PACK
from infrastructure.domain_profile import register_pack_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from tests.fakes import FakeUnitOfWork, FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from datetime import date

    from domain.domain_packs.belarus_law.acts.models import ActVersion, RegulatoryAct


def law_settings() -> DomainSettingsAdapter:
    """Domain settings seeded with the law pack profiles' declared defaults."""
    settings = DomainSettingsAdapter()
    settings.replace_snapshot(
        {
            (default.key, profile.key): default.value
            for profile in BELARUS_LAW_PACK.build_profiles(None)
            for default in profile.config_defaults()
        }
    )
    return settings


def law_registry(settings: DomainSettingsAdapter | None = None) -> DomainProfileRegistry:
    registry = DomainProfileRegistry()
    register_pack_profiles(registry, BELARUS_LAW_PACK, settings or law_settings())
    return registry


class FakeRegulatoryActRepository:
    def __init__(self) -> None:
        self._acts: dict[int, RegulatoryAct] = {}
        self._next_id = 1

    async def find_by_type_and_number(
        self,
        act_type: str,
        act_number: str | None,
        act_date: date | None = None,
        visibility_scope: str = "internal_public",
    ) -> RegulatoryAct | None:
        for act in self._acts.values():
            if (
                act.act_type == act_type
                and act.act_number == act_number
                and act.act_date == act_date
                and act.visibility_scope == visibility_scope
            ):
                return act

        return None

    async def update_act_date(self, act_id: int, act_date: date) -> None:
        self._acts[act_id].act_date = act_date

    async def get_by_id(self, act_id: int) -> RegulatoryAct | None:
        return self._acts.get(act_id)

    async def save(self, act: RegulatoryAct) -> RegulatoryAct:
        act.id = self._next_id
        self._next_id += 1
        self._acts[act.id] = act
        return act

    async def list_all(self) -> list[RegulatoryAct]:
        return list(self._acts.values())


class FakeActVersionRepository:
    def __init__(self) -> None:
        self._versions: dict[int, ActVersion] = {}
        self._next_id = 1

    async def create(self, version: ActVersion) -> ActVersion:
        version = replace(version, id=self._next_id)
        self._next_id += 1
        self._versions[self._next_id - 1] = version
        return version

    async def get_by_id(self, version_id: int) -> ActVersion | None:
        return self._versions.get(version_id)

    async def get_by_document_id(self, document_id: int, *, for_update: bool = False) -> ActVersion | None:
        del for_update

        for v in self._versions.values():
            if v.document_id == document_id:
                return v

        return None

    async def unset_current(self, act_id: int) -> None:
        for v in self._versions.values():
            if v.act_id == act_id and v.is_current and v.id is not None:
                self._versions[v.id] = replace(v, is_current=False)

    async def list_by_act(self, act_id: int, *, for_update: bool = False) -> list[ActVersion]:
        del for_update
        return [v for v in self._versions.values() if v.act_id == act_id]

    async def list_pending_review(self) -> list[ActVersion]:
        return [v for v in self._versions.values() if v.date_source == "extracted" or v.act_id is None]

    async def list_page(self, limit: int, offset: int) -> tuple[list[ActVersion], int]:
        versions = sorted(self._versions.values(), key=lambda item: item.id or 0)
        return versions[offset : offset + limit], len(versions)

    async def update(self, version: ActVersion) -> None:
        if version.id is not None and version.id in self._versions:
            self._versions[version.id] = version


class LawFakeUnitOfWork(FakeUnitOfWork):
    """In-memory unit of work with the law pack's act repositories registered."""

    def __init__(self) -> None:
        super().__init__()
        self.fake_regulatory_acts = self.regulatory_acts = FakeRegulatoryActRepository()
        self.fake_act_versions = self.act_versions = FakeActVersionRepository()
        self._pack_repositories = (ActRepositories(self.regulatory_acts, self.act_versions),)


class LawFakeUnitOfWorkFactory(FakeUnitOfWorkFactory):
    _uow: LawFakeUnitOfWork  # type: ignore[mutable-override]  # Constructed with this subclass only.

    def __init__(self, uow: LawFakeUnitOfWork | None = None) -> None:
        super().__init__(uow or LawFakeUnitOfWork())
