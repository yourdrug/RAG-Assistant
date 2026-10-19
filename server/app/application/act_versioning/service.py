"""Public facade for version planning, commands, review queries and answer invalidation."""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING, Unpack

from application.act_versioning.answers import ActVersionAnswerInvalidator
from application.act_versioning.commands import ActVersionCommands
from application.act_versioning.dto import VersioningResult
from application.act_versioning.planner import ActVersionPlanner
from application.act_versioning.queries import ActVersionQueries


if TYPE_CHECKING:
    from datetime import date

    from application.act_versioning.dto import ActSummary, VersioningPlan, VersionUpdateOptions
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.domain_profile.protocol import DomainProfile, ReferenceMatch
    from domain.domain_profile.settings_port import DomainSettingsPort
    from domain.entities.act_version import ActVersion
    from domain.entities.regulatory_act import RegulatoryAct


log = logging.getLogger("default")


def today() -> date:
    return datetime.now(tz=UTC).astimezone().date()


class ActVersioningService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        settings: DomainSettingsPort,
        cache_invalidator: CacheInvalidatorPort | None = None,
    ) -> None:
        self._planner = ActVersionPlanner(settings)
        self._answers = ActVersionAnswerInvalidator(uow_factory, cache_invalidator)
        self._commands = ActVersionCommands(uow_factory, self._planner, self._answers, today=today)
        self._queries = ActVersionQueries(uow_factory)

    async def find_or_create_act(
        self, act_type: str, extracted_refs: list[ReferenceMatch]
    ) -> RegulatoryAct | None:
        return await self._commands.find_or_create_act(act_type=act_type, extracted_refs=extracted_refs)

    async def handle_versioned_upload(
        self,
        profile: DomainProfile,
        document_id: int,
        extracted_refs: list[ReferenceMatch],
        effective_date: date | None = None,
        date_confidence: float | None = None,
    ) -> ActVersion:
        return await self._commands.handle_versioned_upload(
            profile=profile,
            document_id=document_id,
            extracted_refs=extracted_refs,
            effective_date=effective_date,
            date_confidence=date_confidence,
        )

    async def create_version_in_uow(
        self,
        uow: UnitOfWork,
        profile: DomainProfile,
        document_id: int,
        extracted_refs: list[ReferenceMatch],
        *,
        effective_date: date | None = None,
        date_confidence: float | None = None,
    ) -> ActVersion:
        return await self._commands.create_version_in_uow(
            uow=uow,
            profile=profile,
            document_id=document_id,
            extracted_refs=extracted_refs,
            effective_date=effective_date,
            date_confidence=date_confidence,
        )

    async def invalidate_act_answers(self, act_id: int) -> None:
        return await self._answers.invalidate_act_answers(act_id=act_id)

    async def invalidate_document_answers(self, document_id: int) -> None:
        return await self._answers.invalidate_document_answers(document_id=document_id)

    async def prepare_document_versioning(
        self, profile: DomainProfile, full_text: str
    ) -> VersioningPlan | None:
        return await self._planner.prepare_document_versioning(profile=profile, full_text=full_text)

    async def apply_versioning_plan(
        self,
        uow: UnitOfWork,
        profile: DomainProfile,
        document_id: int,
        plan: VersioningPlan,
    ) -> VersioningResult:
        # The caller owns the transaction and cache invalidation after commit.
        if plan.warning:
            return VersioningResult(plan.domain_metadata, None, None, plan.effective_from, plan.warning)

        version = await self.create_version_in_uow(
            uow,
            profile,
            document_id,
            plan.extracted_refs,
            effective_date=plan.effective_from,
            date_confidence=plan.date_confidence,
        )
        return VersioningResult(
            plan.domain_metadata,
            version.id,
            version.act_id,
            version.effective_from,
            None,
            effective_to=version.effective_to,
            is_current=version.is_current,
        )

    async def list_pending_review(self) -> list[ActVersion]:
        return await self._queries.list_pending_review()

    async def list_versions_page(self, limit: int, offset: int) -> tuple[list[ActVersion], int]:
        return await self._queries.list_versions_page(limit=limit, offset=offset)

    async def list_recent_acts(
        self, limit: int = 20, include_act_ids: set[int] | None = None
    ) -> list[ActSummary]:
        return await self._queries.list_recent_acts(limit=limit, include_act_ids=include_act_ids)

    async def get_document_filenames(self, document_ids: list[int]) -> dict[int, str]:
        return await self._queries.get_document_filenames(document_ids=document_ids)

    async def update_version(
        self,
        version_id: int,
        **options: Unpack[VersionUpdateOptions],
    ) -> None:
        return await self._commands.update_version(
            version_id=version_id,
            **options,
        )

    async def process_document_versioning(
        self,
        profile: DomainProfile,
        document_id: int,
        full_text: str,
    ) -> VersioningResult:
        """Unified versioning entry point for API upload AND CLI ingestion (TZ 8.4).

        Extracts structural references and the effective date, applies the
        trusted-date rule (TZ 9.2: only dates above the domain threshold reach
        the filterable effective_from field), and creates the act version.

        Returns VersioningResult. Versioning is a side concern — on
        failure indexing continues, but the warning must surface in the
        document's warning_message, not vanish.
        """
        empty = VersioningResult(
            domain_metadata=None,
            act_version_id=None,
            act_id=None,
            effective_from=None,
            warning=None,
        )
        plan = await self.prepare_document_versioning(profile, full_text)

        if plan is None:
            return empty

        if plan.warning:
            return VersioningResult(
                domain_metadata=plan.domain_metadata,
                act_version_id=None,
                act_id=None,
                effective_from=plan.effective_from,
                warning=plan.warning,
            )

        try:
            version = await self.handle_versioned_upload(
                profile,
                document_id,
                plan.extracted_refs,
                plan.effective_from,
                plan.date_confidence,
            )
            return VersioningResult(
                domain_metadata=plan.domain_metadata,
                act_version_id=version.id,
                act_id=version.act_id,
                effective_from=version.effective_from,
                warning=None,
                effective_to=version.effective_to,
                is_current=version.is_current,
            )
        except Exception as e:
            log.warning("Versioning failed for doc %d: %s", document_id, e, exc_info=True)
            return VersioningResult(None, None, None, None, f"Версионирование не выполнено: {e}")
