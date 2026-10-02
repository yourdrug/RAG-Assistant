"""Transactional commands for regulatory-act identity and version lifecycle."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime

from domain.domain_profile.date_parsing import parse_date_guess
from domain.domain_profile.protocol import DomainProfile, ReferenceMatch
from domain.entities.act_version import ActVersion
from domain.entities.regulatory_act import RegulatoryAct
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import EntityNotFound, ValidationError
from domain.repositories.chunk_repository import ChunkVersioningRepository
from domain.services.act_version_policy import (
    _document_scope_key,
    _extract_act_number,
    _is_latest_version,
    _new_version_effective_to,
    _same_document_scope,
)

from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.act_version_answers import ActVersionAnswerInvalidator
from application.services.act_version_planner import ActVersionPlanner
from application.services.act_version_timeline import _apply_version_timeline, _relink_version
from application.uow import UnitOfWork

log = logging.getLogger("default")
_DATE_SOURCE_MANUAL = "manual"


class ActVersionCommands:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        planner: ActVersionPlanner,
        answers: ActVersionAnswerInvalidator,
        today: Callable[[], date],
    ) -> None:
        self._uow_factory = uow_factory
        self._planner = planner
        self._answers = answers
        self._today = today

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
        uow: UnitOfWork,
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
                if legacy.id is None:
                    raise RuntimeError("Persisted regulatory act has no ID")
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
            await self._answers.invalidate_act_answers(created.act_id)
        return created

    async def create_version_in_uow(
        self,
        uow: UnitOfWork,
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
        date_source = self._planner.resolve_date_source(profile.key, effective_date, date_confidence)
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
                await _apply_version_timeline(uow, previous, effective_date, today=self._today())

        version = ActVersion(
            id=None,
            act_id=act.id if act is not None else None,
            document_id=document_id,
            effective_from=effective_date,
            effective_to=_new_version_effective_to(previous, effective_date),
            is_current=_is_latest_version(previous, effective_date, today=self._today()),
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
                await _relink_version(uow, version, today=self._today())
            if version.act_id is not None:
                affected_act_ids.add(version.act_id)
            version.date_source = _DATE_SOURCE_MANUAL
            version.verified_by = verified_by
            version.verified_at = datetime.now()

            await uow.act_versions.update(version)
            chunks: ChunkVersioningRepository = uow.chunks
            await chunks.set_current_by_act_version_ids([version_id], version.is_current)

            await chunks.update_temporal_by_act_version_id(
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
            await self._answers.invalidate_act_answers(affected_act_id)
        await self._answers.invalidate_document_answers(version.document_id)
