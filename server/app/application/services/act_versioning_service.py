"""ActVersioningService — manages regulatory act versions and document linkage.

Handles: finding/creating acts by number, version lifecycle (unset_current
plus denormalized chunks sync), trusted-date classification, pending review
queue, and Qdrant metadata sync via the transactional outbox.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from application.dto.versioning_dto import VersioningResult
from domain.domain_profile.settings_port import DomainSettingsPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.domain_profile.protocol import DomainProfile, ReferenceMatch, refs_to_metadata
from domain.entities.act_version import ActVersion
from domain.entities.regulatory_act import RegulatoryAct
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import EntityNotFound

log = logging.getLogger("default")

_ACT_NUMBER_KINDS = ("decree_number", "act_number")
_DATE_SOURCE_EXTRACTED = "extracted"
_DATE_SOURCE_EXTRACTED_TRUSTED = "extracted_trusted"
_DATE_SOURCE_MANUAL = "manual"


class ActVersioningService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        settings: DomainSettingsPort,
    ) -> None:
        self._uow_factory = uow_factory
        self._settings = settings

    async def find_or_create_act(
        self,
        act_type: str,
        extracted_refs: list[ReferenceMatch],
    ) -> RegulatoryAct | None:
        """Find an existing act by number; None when the number is not extracted.

        Without a reliable act number we do NOT guess and do NOT auto-match by
        similarity — the version gets act_id=NULL and lands in the manual
        linkage queue (TZ section 8.3).
        """
        act_number = _extract_act_number(extracted_refs)

        if not act_number:
            return None

        async with self._uow_factory.create(master=True) as uow:
            return await self._find_or_create_act_in_uow(uow, act_type, act_number, extracted_refs)

    @staticmethod
    async def _find_or_create_act_in_uow(
        uow,
        act_type: str,
        act_number: str,
        extracted_refs: list[ReferenceMatch],
    ) -> RegulatoryAct:
        """READ→CHECK→WRITE inside the caller's transaction.

        A unique constraint on (act_type, act_number) backstops the race
        between concurrent uploads of the same act number.
        """
        existing = await uow.regulatory_acts.find_by_type_and_number(act_type, act_number)

        if existing:
            return existing

        title = f"{act_type} №{act_number}"

        for ref in extracted_refs:
            if ref.kind == "decree_date":
                title += f" от {ref.value}"
                break

        return await uow.regulatory_acts.save(
            RegulatoryAct(id=None, act_type=act_type, act_number=act_number, title=title)
        )

    async def handle_versioned_upload(
        self,
        profile: DomainProfile,
        document_id: int,
        extracted_refs: list[ReferenceMatch],
        effective_date: date | None = None,
        date_confidence: float | None = None,
    ) -> ActVersion:
        """Create a new act version for an uploaded document.

        Single transaction: act lookup/create, unsetting the previous current
        version, denormalized chunks sync, new version insert.
        """
        act_number = _extract_act_number(extracted_refs)
        date_source = self._resolve_date_source(profile.key, effective_date, date_confidence)

        async with self._uow_factory.create(master=True) as uow:
            act: RegulatoryAct | None = None
            if act_number:
                act = await self._find_or_create_act_in_uow(uow, profile.key, act_number, extracted_refs)

            if act is not None and act.id is not None:
                previous = await uow.act_versions.list_by_act(act.id)
                previous_ids = [v.id for v in previous if v.is_current and v.id is not None]
                await uow.act_versions.unset_current(act.id)

                # Denormalized chunks of superseded versions stop being "current"
                if previous_ids:
                    await uow.chunks.set_current_by_act_version_ids(previous_ids, False)

            version = ActVersion(
                id=None,
                act_id=act.id if act is not None else None,
                document_id=document_id,
                effective_from=effective_date,
                is_current=True,
                date_source=date_source,
                date_confidence=date_confidence,
            )
            created = await uow.act_versions.create(version)

            if act is None:
                log.info(
                    "Act version %d created without act link (no reliable number) — pending manual linkage",
                    created.id,
                )
            return created

    def _resolve_date_source(
        self,
        domain_key: str,
        effective_date: date | None,
        date_confidence: float | None,
    ) -> str:
        if effective_date is None:
            return _DATE_SOURCE_EXTRACTED
        # Seeding guarantees the key exists — a missing key is a loud config error.
        threshold = float(self._settings.get("effective_date_auto_trust_threshold", domain_key=domain_key))
        if date_confidence is not None and date_confidence >= threshold:
            return _DATE_SOURCE_EXTRACTED_TRUSTED
        return _DATE_SOURCE_EXTRACTED

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
        if not profile.is_versioned:
            return empty
        try:
            refs = profile.extract_references(full_text)
            domain_metadata = refs_to_metadata(refs) if refs else None

            date_candidate = profile.extract_effective_date(full_text)
            raw_effective_date = (
                date_candidate.effective_from if date_candidate and date_candidate.effective_from else None
            )
            effective_from = None
            if raw_effective_date is not None:
                threshold = float(
                    self._settings.get("effective_date_auto_trust_threshold", domain_key=profile.key)
                )
                if date_candidate is not None and date_candidate.confidence >= threshold:
                    effective_from = raw_effective_date

            version = await self.handle_versioned_upload(
                profile=profile,
                document_id=document_id,
                extracted_refs=refs or [],
                effective_date=raw_effective_date,
                date_confidence=date_candidate.confidence if date_candidate else None,
            )
            return VersioningResult(
                domain_metadata=domain_metadata,
                act_version_id=version.id,
                act_id=version.act_id,
                effective_from=effective_from,
                warning=None,
            )
        except Exception as e:
            log.warning("Versioning failed for doc %d: %s", document_id, e)
            return VersioningResult(
                domain_metadata=None,
                act_version_id=None,
                act_id=None,
                effective_from=None,
                warning=f"Версионирование не выполнено: {e}",
            )

    async def list_pending_review(self) -> list[ActVersion]:
        """List versions needing manual review (not trusted, not linked)."""
        async with self._uow_factory.create() as uow:
            return await uow.act_versions.list_pending_review()

    async def list_recent_acts(self, limit: int = 20) -> list:
        """Recent acts — linkage suggestions for versions pending manual review."""
        from dataclasses import dataclass

        @dataclass
        class ActSummary:
            id: int
            act_type: str
            act_number: str | None
            title: str

        async with self._uow_factory.create() as uow:
            acts = await uow.regulatory_acts.list_all()
        return [
            ActSummary(id=a.id, act_type=a.act_type, act_number=a.act_number or "", title=a.title)
            for a in acts[-limit:]
            if a.id is not None
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

    async def update_version(
        self,
        version_id: int,
        *,
        effective_from: date | None = None,
        effective_to: date | None = None,
        act_id: int | None = None,
        verified_by: int | None = None,
    ) -> None:
        """Update version dates/linkage via admin review. Always sets date_source='manual'.

        Single transaction: act_versions row + denormalized chunks in Postgres
        + outbox entry that patches Qdrant payload (no re-embedding needed).
        """
        async with self._uow_factory.create(master=True) as uow:
            version = await uow.act_versions.get_by_id(version_id)
            if version is None:
                raise EntityNotFound("ActVersion", version_id)

            if effective_from is not None:
                version.effective_from = effective_from
            if effective_to is not None:
                version.effective_to = effective_to
            if act_id is not None:
                version.act_id = act_id
            version.date_source = _DATE_SOURCE_MANUAL
            version.verified_by = verified_by
            version.verified_at = datetime.now()

            await uow.act_versions.update(version)

            await uow.chunks.update_temporal_by_act_version_id(
                version_id,
                effective_from=version.effective_from,
                effective_to=version.effective_to,
            )

            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.UPDATE_METADATA,
                    aggregate_type="act_version",
                    aggregate_id=version_id,
                    payload={
                        "act_version_id": version_id,
                        "effective_from": (
                            version.effective_from.isoformat() if version.effective_from else None
                        ),
                        "effective_to": version.effective_to.isoformat() if version.effective_to else None,
                    },
                )
            )


def _extract_act_number(extracted_refs: list[ReferenceMatch]) -> str | None:
    for ref in extracted_refs:
        if ref.kind in _ACT_NUMBER_KINDS:
            return ref.value
    return None
