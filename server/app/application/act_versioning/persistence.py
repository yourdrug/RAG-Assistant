"""Transactional loading and persistence of act versions and search projections."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from domain.knowledge.acts.edition_metadata import edition_metadata
from domain.knowledge.acts.timeline import ActVersionTimeline
from domain.knowledge.index_sync import OutboxOperation, VectorOutboxEntry
from domain.shared.exceptions import EntityNotFoundError


if TYPE_CHECKING:
    from application.uow import UnitOfWork
    from domain.knowledge.acts.models import ActVersion
    from domain.knowledge.repositories.chunk import ChunkPackMetadataRepository


class ActVersionPersistence:
    def __init__(self, uow: UnitOfWork) -> None:
        self.uow = uow

    async def load(self, version_id: int) -> tuple[ActVersion, ActVersionTimeline]:
        version = await self.uow.act_versions.get_by_id(version_id)

        if version is None:
            entity = "ActVersion"
            raise EntityNotFoundError(entity, version_id)

        timeline = ActVersionTimeline()

        if version.act_id is not None:
            versions = await self.uow.act_versions.list_by_act(version.act_id, for_update=True)
            timeline = ActVersionTimeline(tuple(versions))
            # Resolve from the snapshot read after acquiring the aggregate lock.
            version = timeline.get_version(version_id)

        return replace(version), timeline

    async def save_changes(
        self, timeline: ActVersionTimeline, versions: tuple[ActVersion, ...], *, edited_id: int | None = None
    ) -> None:
        # Answers citing any edition of the act may change when one edition moves.
        related = sorted({version.document_id for version in (*timeline.versions, *versions)})

        for version in timeline.changed_versions(versions, edited_id=edited_id):
            await self.save(version, related_document_ids=related)

    async def save(self, version: ActVersion, *, related_document_ids: list[int] | None = None) -> None:
        # Persist a version and both search projections in one transaction.
        if version.id is None:
            message = "Persisted act version has no ID"
            raise RuntimeError(message)

        await self.uow.act_versions.update(version)
        chunks: ChunkPackMetadataRepository = self.uow.chunks
        await chunks.set_pack_metadata(version.document_id, edition_metadata(version))
        await self.uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.UPDATE_METADATA,
                aggregate_type="document",
                aggregate_id=version.document_id,
                payload={"related_document_ids": related_document_ids or [version.document_id]},
            )
        )
