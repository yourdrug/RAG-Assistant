"""Update version rows, chunk projections and vector outbox in the caller's transaction."""

from __future__ import annotations

from datetime import date

from domain.entities.act_version import ActVersion
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import EntityNotFound, ValidationError
from domain.repositories.chunk_repository import ChunkVersioningRepository
from domain.services.act_version_policy import (
    _is_latest_version,
    _new_version_effective_to,
    _same_document_scope,
)

from application.uow import UnitOfWork


async def _relink_version(uow: UnitOfWork, version: ActVersion, *, today: date) -> None:
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
        await _apply_version_timeline(uow, previous, version.effective_from, today=today)
        version.is_current = _is_latest_version(previous, version.effective_from, today=today)
        if version.effective_to is None:
            version.effective_to = _new_version_effective_to(previous, version.effective_from)


async def _apply_version_timeline(
    uow: UnitOfWork, previous: list[ActVersion], new_date: date | None, *, today: date
) -> None:
    """Close the preceding interval and place backfilled versions in date order."""
    if new_date is None:
        return

    chunks: ChunkVersioningRepository = uow.chunks
    new_is_current = _is_latest_version(previous, new_date, today=today)

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
            await chunks.set_current_by_act_version_ids([version.id], version.is_current)
            await chunks.update_temporal_by_act_version_id(
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
