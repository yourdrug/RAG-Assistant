"""Rename command: copy storage object and update document/chunks/outbox atomically."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from application.documents.dto import DocumentDTO
from application.documents.utils import generate_storage_key, resolve_unique_filename
from domain.identity.roles import UserKind
from domain.knowledge.access_control import check_ownership
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import OutboxOperation, VectorOutboxEntry
from domain.shared.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError


if TYPE_CHECKING:
    from application.identity import UserContextFactory
    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWork, UnitOfWorkFactory


log = logging.getLogger("default")


class DocumentRenameCommand:
    """Own rename transaction and compensation for a copied storage object."""

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
        self,
        document_id: int,
        new_filename: str,
        user_id: int,
        user_role: str,
        user_kind: str = UserKind.INTERNAL,
    ) -> DocumentDTO:
        ext = Path(new_filename).suffix.lower()

        if ext not in self._file_storage.supported_extensions:
            msg = f"Unsupported file format: {ext}"
            raise ValidationError(msg)

        copied_paths: list[str] = []
        try:
            async with self._uow_factory.create(master=True) as uow:
                dto, old_source_path = await self.rename_in_transaction(
                    uow,
                    document_id,
                    new_filename,
                    user_id,
                    user_role=user_role,
                    user_kind=user_kind,
                    copied_paths=copied_paths,
                )
        except BaseException:
            if copied_paths:
                try:
                    await self._file_storage.delete_file(copied_paths[0])
                except Exception:
                    log.exception(
                        "Failed to clean up orphaned copy %s for document %d", copied_paths[0], document_id
                    )

            raise

        if old_source_path:
            try:
                await self._file_storage.delete_file(old_source_path)
            except Exception:
                log.exception(
                    "Failed to delete old storage object %s after rename of document %d — orphaned",
                    old_source_path,
                    document_id,
                )

        return dto

    async def rename_in_transaction(
        self,
        uow: UnitOfWork,
        document_id: int,
        new_filename: str,
        user_id: int,
        *,
        user_role: str,
        user_kind: str,
        copied_paths: list[str],
    ) -> tuple[DocumentDTO, str | None]:
        doc = await uow.documents.get_by_id(document_id)

        if doc is None:
            msg = "Document"
            raise EntityNotFound(msg, document_id)

        ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
        check_ownership(doc, ctx, "rename")
        owner_id, effective_group_id = doc.owner_id, doc.group_id
        existing = await uow.documents.find_active_slot(
            owner_id, new_filename, effective_group_id, for_update=True
        )

        if existing and existing.id != document_id:
            if existing.status in (
                DocumentStatus.PENDING,
                DocumentStatus.PROCESSING,
                DocumentStatus.INDEXING,
            ):
                msg = "This document name is already being processed"
                raise BusinessRuleViolation(msg)

            if existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
                new_filename = await resolve_unique_filename(
                    uow.documents, owner_id, effective_group_id, new_filename
                )

        # Renaming an assignment must not move the shared ingestion source.
        new_source_path = (
            doc.source_path
            if doc.ingestion_key is not None
            else generate_storage_key(owner_id, effective_group_id, document_id, new_filename)
        )
        old_source_path: str | None = None

        if doc.source_path and doc.source_path != new_source_path:
            await self._file_storage.copy_file(doc.source_path, new_source_path)
            copied_paths.append(new_source_path)
            old_source_path = doc.source_path

        await uow.documents.update_filename(document_id, new_filename, new_source_path)
        await uow.chunks.update_filename_by_document_id(document_id, new_filename)
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.RENAME_DOCUMENT,
                aggregate_type="document",
                aggregate_id=document_id,
                payload={"document_id": document_id, "filename": new_filename},
            )
        )
        final_doc = await uow.documents.get_by_id(document_id)

        if final_doc is None:
            msg = "Document"
            raise EntityNotFound(msg, document_id)

        return DocumentDTO.from_entity(final_doc), old_source_path
