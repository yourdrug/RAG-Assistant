"""Answer sources of channel documents: the cited file in the revision that was published."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.rag.dto import StorageSource


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory


class SyncSourceLinks:
    """Link to the cited version, not to the branch: an old answer keeps opening what it cited."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def describe(self, document_ids: list[int]) -> dict[int, StorageSource]:
        if not document_ids:
            return {}

        async with self._uow_factory.create() as uow:
            documents = [
                doc
                for doc in await uow.documents.get_by_ids(document_ids)
                if doc.sync and doc.sync.storage_revision
            ]
            channels = {
                channel_id: await uow.sync_channels.get_by_id(channel_id)
                for channel_id in {doc.sync.channel_id for doc in documents if doc.sync}
            }

        sources = {}

        for doc in documents:
            channel = channels.get(doc.sync.channel_id) if doc.sync else None

            if doc.id is None or doc.sync is None or doc.sync.storage_revision is None or channel is None:
                continue

            location = channel.settings.location
            sources[doc.id] = StorageSource(
                branch=location.branch,
                path=doc.sync.path,
                revision=doc.sync.storage_revision,
                url=location.file_url(doc.sync.storage_revision, doc.sync.path),
            )

        return sources
