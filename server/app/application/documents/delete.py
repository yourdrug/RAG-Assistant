"""Delete command: authorize, enqueue vector deletion and clean up storage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.indexing import enqueue_delete_by_document
from domain.exceptions import EntityNotFound
from domain.services import check_ownership
from domain.value_objects.roles import UserKind


if TYPE_CHECKING:
    from application.identity import UserContextFactory
    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWorkFactory


log = logging.getLogger("default")


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
                raise EntityNotFound(msg, document_id)

            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_ownership(doc, ctx, "delete")
            await enqueue_delete_by_document(uow, document_id)
            source_path = doc.source_path
            await uow.conversations.clear_summaries_referencing(document_id)
            await uow.documents.delete(document_id)

        if source_path:
            try:
                await self._file_storage.delete_file(source_path)
            except Exception:
                log.exception(
                    "Failed to delete storage object %s for document %d — orphaned", source_path, document_id
                )
