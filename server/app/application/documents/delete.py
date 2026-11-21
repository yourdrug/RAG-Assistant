"""Delete command: authorize, enqueue vector deletion and clean up storage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.indexing import enqueue_delete_by_document
from domain.identity.roles import UserKind
from domain.knowledge.access_control import check_ownership
from domain.shared.exceptions import EntityNotFoundError


if TYPE_CHECKING:
    from application.identity import UserContextFactory
    from application.indexing.pack_hooks import DocumentHooksResolver, DocumentPackHooks
    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWork, UnitOfWorkFactory


log = logging.getLogger("default")


async def delete_document_in_uow(
    uow: UnitOfWork, document_id: int, pack_hooks: DocumentPackHooks | None = None
) -> None:
    """Delete a document; the pack keeps its own state consistent in the same transaction."""

    async def delete() -> None:
        await enqueue_delete_by_document(uow, document_id)
        await uow.conversations.clear_summaries_referencing(document_id)
        await uow.documents.delete(document_id)

    if pack_hooks is None:
        await delete()
    else:
        await pack_hooks.delete_document(uow, document_id, delete)


class DocumentDeleteCommand:
    """Own document deletion and storage cleanup after the transaction commits."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        file_storage: FileStorage,
        user_ctx_factory: UserContextFactory,
        pack_hooks: DocumentHooksResolver | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._user_ctx_factory = user_ctx_factory
        self._pack_hooks = pack_hooks

    async def execute(
        self, document_id: int, user_id: int, user_role: str, user_kind: str = UserKind.INTERNAL
    ) -> None:
        async with self._uow_factory.create(master=True) as uow:
            # Locked: a channel adopting the document meanwhile must not be overtaken.
            doc = await uow.documents.get_by_id(document_id, for_update=True)

            if doc is None:
                msg = "Document"
                raise EntityNotFoundError(msg, document_id)

            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_ownership(doc, ctx, "delete")
            doc.ensure_manually_managed("delete")
            # Ingestion owns shared source objects, rather than individual assignments.
            source_path = doc.source_path if doc.ingestion_key is None else None
            hooks = await self._pack_hooks.for_document(uow, doc) if self._pack_hooks is not None else None
            await delete_document_in_uow(uow, document_id, hooks)

        if source_path:
            try:
                await self._file_storage.delete_file(source_path)
            except Exception:
                log.exception(
                    "Failed to delete storage object %s for document %d — orphaned", source_path, document_id
                )
