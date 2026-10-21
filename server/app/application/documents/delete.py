"""Delete command: authorize, enqueue vector deletion and clean up storage."""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING

from application.act_versioning.persistence import ActVersionPersistence
from application.indexing import enqueue_delete_by_document
from domain.identity.roles import UserKind
from domain.knowledge.access_control import check_ownership
from domain.shared.exceptions import EntityNotFoundError


if TYPE_CHECKING:
    from application.identity import UserContextFactory
    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWork, UnitOfWorkFactory


log = logging.getLogger("default")


async def delete_document_in_uow(uow: UnitOfWork, document_id: int) -> None:
    """Delete a document and reconnect its act timeline in the same transaction."""
    version = await uow.act_versions.get_by_document_id(document_id)
    persistence = ActVersionPersistence(uow)
    timeline = None

    if version is not None and version.act_id is not None and version.id is not None:
        version, timeline = await persistence.load(version.id)

    await enqueue_delete_by_document(uow, document_id)
    await uow.conversations.clear_summaries_referencing(document_id)
    # The cascade removes the current edition before another can be promoted.
    await uow.documents.delete(document_id)

    if timeline is not None and version is not None and version.id is not None:
        remaining = timeline.remove(version.id, today=datetime.now(UTC).astimezone().date())
        await persistence.save_changes(timeline, remaining.versions)


class DocumentDeleteCommand:
    """Own document deletion and storage cleanup after the transaction commits."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        file_storage: FileStorage,
        user_ctx_factory: UserContextFactory,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._user_ctx_factory = user_ctx_factory

    async def execute(
        self, document_id: int, user_id: int, user_role: str, user_kind: str = UserKind.INTERNAL
    ) -> None:
        async with self._uow_factory.create(master=True) as uow:
            doc = await uow.documents.get_by_id(document_id)

            if doc is None:
                msg = "Document"
                raise EntityNotFoundError(msg, document_id)

            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_ownership(doc, ctx, "delete")
            # Ingestion owns shared source objects, rather than individual assignments.
            source_path = doc.source_path if doc.ingestion_key is None else None
            await delete_document_in_uow(uow, document_id)

        if source_path:
            try:
                await self._file_storage.delete_file(source_path)
            except Exception:
                log.exception(
                    "Failed to delete storage object %s for document %d — orphaned", source_path, document_id
                )
