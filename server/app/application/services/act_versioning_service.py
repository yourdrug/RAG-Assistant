"""ActVersioningService — manages regulatory act versions and document linkage.

Handles: finding/creating acts by number, version lifecycle (unset_current
plus denormalized chunks sync), trusted-date classification, pending review
queue, and Qdrant metadata sync via the transactional outbox.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from application.dto.versioning_dto import VersioningPlan, VersioningResult
from application.ports.cache_invalidator import CacheInvalidatorPort
from domain.domain_profile.settings_port import DomainSettingsPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.domain_profile.protocol import DomainProfile, ReferenceMatch, refs_to_metadata
from domain.domain_profile.date_parsing import parse_date_guess
from domain.entities.act_version import ActVersion
from domain.entities.regulatory_act import RegulatoryAct
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import EntityNotFound
from domain.exceptions import ValidationError

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
        cache_invalidator: CacheInvalidatorPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._settings = settings
        self._cache_invalidator = cache_invalidator

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
        visibility_scope: str = "internal_public",
    ) -> RegulatoryAct:
        """READ→CHECK→WRITE inside the caller's transaction.

        A unique constraint on (act_type, act_number) backstops the race
        between concurrent uploads of the same act number.
        """
        signing_date = next(
            (
                parse_date_guess(ref.value)
                for ref in extracted_refs
                if ref.kind in ("decree_date", "act_date")
            ),
            None,
        )
        existing = await uow.regulatory_acts.find_by_type_and_number(
            act_type, act_number, signing_date, visibility_scope
        )
        if existing is None and signing_date is not None:
            legacy = await uow.regulatory_acts.find_by_type_and_number(
                act_type, act_number, None, visibility_scope
            )
            if legacy is not None and parse_date_guess(legacy.title) == signing_date:
                await uow.regulatory_acts.update_act_date(legacy.id, signing_date)
                legacy.act_date = signing_date
                existing = legacy

        if existing:
            return existing

        title = f"{act_type} №{act_number}"

        for ref in extracted_refs:
            if ref.kind in ("decree_date", "act_date"):
                title += f" от {ref.value}"
                break

        return await uow.regulatory_acts.save(
            RegulatoryAct(
                id=None,
                act_type=act_type,
                act_number=act_number,
                title=title,
                act_date=signing_date,
                visibility_scope=visibility_scope,
            )
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
        async with self._uow_factory.create(master=True) as uow:
            created = await self.create_version_in_uow(
                uow, profile, document_id, extracted_refs, effective_date, date_confidence
            )
        if created.act_id is not None:
            await self.invalidate_act_answers(created.act_id)
        return created

    async def invalidate_act_answers(self, act_id: int) -> None:
        if self._cache_invalidator is None:
            return
        async with self._uow_factory.create(master=True) as uow:
            versions = await uow.act_versions.list_by_act(act_id)
            document_ids = list({v.document_id for v in versions})
        if document_ids:
            await self._cache_invalidator.invalidate_by_document_ids(document_ids)

    async def invalidate_document_answers(self, document_id: int) -> None:
        if self._cache_invalidator is not None:
            await self._cache_invalidator.invalidate_by_document_ids([document_id])

    async def create_version_in_uow(
        self,
        uow,
        profile: DomainProfile,
        document_id: int,
        extracted_refs: list[ReferenceMatch],
        effective_date: date | None = None,
        date_confidence: float | None = None,
    ) -> ActVersion:
        """Create a version using the caller's transaction."""
        existing_version = await uow.act_versions.get_by_document_id(document_id, for_update=True)
        if existing_version is not None:
            return existing_version
        act_number = _extract_act_number(extracted_refs)
        date_source = self._resolve_date_source(profile.key, effective_date, date_confidence)
        act: RegulatoryAct | None = None
        previous: list[ActVersion] = []
        if act_number:
            document = await uow.documents.get_by_id(document_id)
            scope = _document_scope_key(document) if document is not None else "internal_public"
            act = await self._find_or_create_act_in_uow(uow, profile.key, act_number, extracted_refs, scope)

        if act is not None and act.id is not None:
            previous = await uow.act_versions.list_by_act(act.id)
            new_doc = await uow.documents.get_by_id(document_id)
            previous_docs = [await uow.documents.get_by_id(v.document_id) for v in previous]
            same_scope = new_doc is not None and all(
                old_doc is not None and _same_document_scope(new_doc, old_doc) for old_doc in previous_docs
            )
            if previous and not same_scope:
                # A version uploaded into another visibility scope must never
                # change the current flag of a public/private/group act.
                log.warning(
                    "Act number %s already exists in another visibility scope; leaving new version unlinked",
                    act_number,
                )
                act = None
                previous = []
            elif previous:
                await _apply_version_timeline(uow, previous, effective_date)

        version = ActVersion(
            id=None,
            act_id=act.id if act is not None else None,
            document_id=document_id,
            effective_from=effective_date,
            effective_to=_new_version_effective_to(previous, effective_date),
            is_current=_is_latest_version(previous, effective_date),
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
            log.warning("Versioning failed for doc %d: %s", document_id, e)
            return VersioningResult(None, None, None, None, f"Версионирование не выполнено: {e}")

    async def prepare_document_versioning(
        self, profile: DomainProfile, full_text: str
    ) -> VersioningPlan | None:
        """Extract version metadata without committing a version row."""
        if not profile.is_versioned:
            return None
        try:
            refs = profile.extract_references(full_text) or []
            date_candidate = profile.extract_effective_date(full_text)
            raw_effective_date = (
                date_candidate.effective_from if date_candidate and date_candidate.effective_from else None
            )
            effective_from = None
            if raw_effective_date is not None and date_candidate is not None:
                threshold = float(
                    self._settings.get("effective_date_auto_trust_threshold", domain_key=profile.key)
                )
                if date_candidate.confidence >= threshold:
                    effective_from = raw_effective_date
            return VersioningPlan(
                extracted_refs=refs,
                domain_metadata=refs_to_metadata(refs) if refs else None,
                effective_date=raw_effective_date,
                effective_from=effective_from,
                date_confidence=date_candidate.confidence if date_candidate else None,
            )
        except Exception as e:
            log.warning("Versioning extraction failed for domain %s: %s", profile.key, e)
            return VersioningPlan([], None, None, None, None, f"Версионирование не выполнено: {e}")

    async def list_pending_review(self) -> list[ActVersion]:
        """List versions needing manual review (not trusted, not linked)."""
        async with self._uow_factory.create() as uow:
            return await uow.act_versions.list_pending_review()

    async def list_versions_page(self, limit: int, offset: int) -> tuple[list[ActVersion], int]:
        async with self._uow_factory.create() as uow:
            return await uow.act_versions.list_page(limit, offset)

    async def list_recent_acts(self, limit: int = 20, include_act_ids: set[int] | None = None) -> list:
        """Recent acts — linkage suggestions for versions pending manual review."""
        from dataclasses import dataclass

        @dataclass
        class ActSummary:
            id: int
            act_type: str
            act_number: str | None
            title: str
            act_date: date | None
            visibility_scope: str

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

    async def update_version(
        self,
        version_id: int,
        *,
        effective_from: date | None = None,
        effective_to: date | None = None,
        act_id: int | None = None,
        verified_by: int | None = None,
        effective_from_provided: bool = False,
        effective_to_provided: bool = False,
        act_id_provided: bool = False,
    ) -> None:
        """Update version dates/linkage via admin review. Always sets date_source='manual'.

        Single transaction: act_versions row + denormalized chunks in Postgres
        + outbox entry that patches Qdrant payload (no re-embedding needed).
        """
        affected_act_ids: set[int] = set()
        async with self._uow_factory.create(master=True) as uow:
            version = await uow.act_versions.get_by_id(version_id)
            if version is None:
                raise EntityNotFound("ActVersion", version_id)

            old_act_id = version.act_id
            if old_act_id is not None:
                affected_act_ids.add(old_act_id)
            if effective_from_provided or effective_from is not None:
                version.effective_from = effective_from
            if effective_to_provided or effective_to is not None:
                version.effective_to = effective_to
            if act_id_provided or act_id is not None:
                version.act_id = act_id
            if (
                version.effective_from is not None
                and version.effective_to is not None
                and version.effective_to <= version.effective_from
            ):
                raise ValidationError("effective_to must be later than effective_from")
            if old_act_id != version.act_id:
                await _relink_version(uow, version)
            if version.act_id is not None:
                affected_act_ids.add(version.act_id)
            version.date_source = _DATE_SOURCE_MANUAL
            version.verified_by = verified_by
            version.verified_at = datetime.now()

            await uow.act_versions.update(version)
            await uow.chunks.set_current_by_act_version_ids([version_id], version.is_current)

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
                        "act_id": version.act_id,
                        "is_current": version.is_current,
                        "effective_from": (
                            version.effective_from.isoformat() if version.effective_from else None
                        ),
                        "effective_to": version.effective_to.isoformat() if version.effective_to else None,
                    },
                )
            )

        for affected_act_id in affected_act_ids:
            await self.invalidate_act_answers(affected_act_id)
        await self.invalidate_document_answers(version.document_id)


def _extract_act_number(extracted_refs: list[ReferenceMatch]) -> str | None:
    for ref in extracted_refs:
        if ref.kind in _ACT_NUMBER_KINDS:
            return ref.value
    return None


def _same_document_scope(left, right) -> bool:
    if left.visibility != right.visibility:
        return False
    if left.visibility in ("internal_private", "client_private"):
        return left.owner_id == right.owner_id
    if left.visibility == "internal_group":
        return left.group_id == right.group_id
    return left.visibility == "internal_public"


def _document_scope_key(document) -> str:
    if document.visibility == "internal_group":
        return f"{document.visibility}:{document.group_id}"
    if document.visibility in ("internal_private", "client_private"):
        return f"{document.visibility}:{document.owner_id}"
    return str(document.visibility)


async def _relink_version(uow, version: ActVersion) -> None:
    if version.act_id is not None:
        target_act = await uow.regulatory_acts.get_by_id(version.act_id)
        if target_act is None:
            raise EntityNotFound("RegulatoryAct", version.act_id)
        previous = await uow.act_versions.list_by_act(version.act_id)
        previous = [item for item in previous if item.id != version.id]
        document = await uow.documents.get_by_id(version.document_id)
        for item in previous:
            other = await uow.documents.get_by_id(item.document_id)
            if document is None or other is None or not _same_document_scope(document, other):
                raise ValidationError("Versions of an act must have the same visibility scope")
        await _apply_version_timeline(uow, previous, version.effective_from)
        version.is_current = _is_latest_version(previous, version.effective_from)
        if version.effective_to is None:
            version.effective_to = _new_version_effective_to(previous, version.effective_from)


def _is_latest_version(previous: list[ActVersion], effective_date: date | None) -> bool:
    if effective_date is None:
        return not any(v.is_current for v in previous)
    if effective_date > date.today():
        return False
    current_dates = [
        v.effective_from
        for v in previous
        if v.effective_from is not None and v.effective_from <= date.today()
    ]
    return not current_dates or effective_date >= max(current_dates)


async def _apply_version_timeline(uow, previous: list[ActVersion], new_date: date | None) -> None:
    """Close the preceding interval and place backfilled versions in date order."""
    if new_date is None:
        return

    new_is_current = _is_latest_version(previous, new_date)

    for version in previous:
        changed = False
        # A corrected import with the same start date supersedes the earlier
        # edition completely; its old interval becomes empty.
        if version.effective_from is None or version.effective_from <= new_date:
            if version.effective_to is None or version.effective_to > new_date:
                version.effective_to = new_date
                changed = True
        if new_is_current and version.is_current:
            version.is_current = False
            changed = True

        if changed and version.id is not None:
            await uow.act_versions.update(version)
            await uow.chunks.set_current_by_act_version_ids([version.id], version.is_current)
            await uow.chunks.update_temporal_by_act_version_id(
                version.id,
                effective_from=version.effective_from,
                effective_to=version.effective_to,
            )
            payload = {
                "act_version_id": version.id,
                "is_current": version.is_current,
                "effective_from": version.effective_from.isoformat() if version.effective_from else None,
                "effective_to": version.effective_to.isoformat() if version.effective_to else None,
            }
            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.UPDATE_METADATA,
                    aggregate_type="act_version",
                    aggregate_id=version.id,
                    payload=payload,
                )
            )


def _new_version_effective_to(previous: list[ActVersion], effective_date: date | None) -> date | None:
    if effective_date is None:
        return None
    later_dates = [
        v.effective_from for v in previous if v.effective_from and v.effective_from > effective_date
    ]
    return min(later_dates) if later_dates else None
