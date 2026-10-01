"""Read models for regulatory-act version review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from domain.entities.act_version import ActVersion

from application.ports.unit_of_work_factory import UnitOfWorkFactory


@dataclass
class ActSummary:
    id: int
    act_type: str
    act_number: str | None
    title: str
    act_date: date | None
    visibility_scope: str


class ActVersionQueries:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list_pending_review(self) -> list[ActVersion]:
        """List versions needing manual review (not trusted, not linked)."""
        async with self._uow_factory.create() as uow:
            return await uow.act_versions.list_pending_review()

    async def list_versions_page(self, limit: int, offset: int) -> tuple[list[ActVersion], int]:
        async with self._uow_factory.create() as uow:
            return await uow.act_versions.list_page(limit, offset)

    async def list_recent_acts(
        self, limit: int = 20, include_act_ids: set[int] | None = None
    ) -> list[ActSummary]:
        """Recent acts — linkage suggestions for versions pending manual review."""
        async with self._uow_factory.create() as uow:
            acts = await uow.regulatory_acts.list_all()
        return [
            ActSummary(
                id=a.id,
                act_type=a.act_type,
                act_number=a.act_number or "",
                title=a.title,
                act_date=a.act_date,
                visibility_scope=a.visibility_scope,
            )
            for a in acts
            if a.id is not None and (a in acts[-limit:] or a.id in (include_act_ids or set()))
        ]

    async def get_document_filenames(self, document_ids: list[int]) -> dict[int, str]:
        """Fetch filenames for a list of document IDs (for display in admin UI)."""
        async with self._uow_factory.create() as uow:
            result: dict[int, str] = {}
            for did in document_ids:
                doc = await uow.documents.get_by_id(did)
                if doc is not None:
                    result[did] = doc.filename
            return result
