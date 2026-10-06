"""Transactional loading and persistence of act versions and search projections."""

from __future__ import annotations

from dataclasses import replace

from domain.entities.act_version import ActVersion
from domain.entities.act_version_timeline import ActVersionTimeline
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import EntityNotFound
from domain.repositories.chunk_repository import ChunkVersioningRepository

from application.uow import UnitOfWork


class ActVersionPersistence:
    def __init__(self, uow: UnitOfWork) -> None:
        self.uow = uow

    async def load(self, version_id: int) -> tuple[ActVersion, ActVersionTimeline]:
        version = await self.uow.act_versions.get_by_id(version_id)
        if version is None:
            raise EntityNotFound("ActVersion", version_id)
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
        for version in timeline.changed_versions(versions, edited_id=edited_id):
            await self.save(version)

    async def save(self, version: ActVersion) -> None:
        """Persist a version and both search projections in one transaction."""
        if version.id is None:
            raise RuntimeError("Persisted act version has no ID")
        await self.uow.act_versions.update(version)
        chunks: ChunkVersioningRepository = self.uow.chunks
        await chunks.set_current_by_act_version_ids([version.id], version.is_current)
        await chunks.update_temporal_by_act_version_id(
            version.id, effective_from=version.effective_from, effective_to=version.effective_to
        )
        await self.uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.UPDATE_METADATA,
                aggregate_type="act_version",
                aggregate_id=version.id,
                payload={
                    "act_version_id": version.id,
                    "act_id": version.act_id,
                    "is_current": version.is_current,
                    "effective_from": version.effective_from.isoformat() if version.effective_from else None,
                    "effective_to": version.effective_to.isoformat() if version.effective_to else None,
                },
            )
        )
