"""Upload command: validate scope, reserve a name, store and resolve replacement."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

from application.documents.conflicts import (
    resolve_upload_conflict,
    resolve_upload_version_group,
)
from application.documents.dto import DocumentDTO
from application.documents.utils import generate_storage_key, resolve_unique_filename
from domain.entities.document import Document
from domain.exceptions import (
    BusinessRuleViolation,
    EntityNotFound,
    UniqueConstraintViolation,
    ValidationError,
)
from domain.services import compute_owner_and_group, validate_document_visibility
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility


if TYPE_CHECKING:
    from application.documents.dto import UploadOptions
    from application.identity import UserContextFactory
    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.value_objects.user_context import UserContext

log = logging.getLogger("default")


class DocumentUploadCommand:
    """Own the upload transaction and compensation for a failed storage write."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        file_storage: FileStorage,
        user_ctx_factory: UserContextFactory,
        domain_registry: DomainProfileRegistry | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._user_ctx_factory = user_ctx_factory
        self._domain_registry = domain_registry

    async def resolve_effective_owner_id(
        self,
        uow: UnitOfWork,
        vis: DocumentVisibility,
        user_id: int,
        user_kind: str,
        *,
        client_id: int | None,
        user_role: str = UserRole.USER,
        managed_client_ids: tuple[int, ...] | None = None,
    ) -> int:
        if vis != DocumentVisibility.CLIENT_PRIVATE:
            return user_id

        if user_kind == UserKind.CLIENT:
            return user_id

        if client_id is None:
            msg = "client_id required for client_private upload"
            raise ValidationError(msg)

        client_user = await uow.users.get_by_id(client_id)

        if client_user is None or client_user.kind != UserKind.CLIENT:
            msg = "client_id must be a user with kind='client'"
            raise ValidationError(msg)

        if user_role == UserRole.CURATOR and (
            managed_client_ids is None or client_id not in managed_client_ids
        ):
            msg = "You can only upload documents for assigned clients"
            raise BusinessRuleViolation(msg)

        return client_id

    @staticmethod
    def validate_no_active_processing(existing: Document | None) -> None:
        if existing and existing.status in (
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
            DocumentStatus.INDEXING,
        ):
            msg = "This document is already being processed"
            raise BusinessRuleViolation(msg)

    async def persist_upload(
        self,
        uow: UnitOfWork,
        doc: Document,
        file_data: bytes,
        owner_id: int | None,
        *,
        effective_group_id: int | None,
        filename: str,
    ) -> tuple[Document, str]:
        try:
            saved_doc = await uow.documents.save(doc)
        except UniqueConstraintViolation as exc:
            msg = "This document is already being uploaded by a concurrent request"
            raise BusinessRuleViolation(msg) from exc

        if saved_doc.id is None:
            msg = "Document save returned None id"
            raise RuntimeError(msg)

        key = generate_storage_key(owner_id, effective_group_id, saved_doc.id, filename)
        try:
            await self._file_storage.upload_file(key, file_data)
            await uow.documents.set_source_path(saved_doc.id, key)
        except BaseException:
            try:
                await self._file_storage.delete_file(key)
            except Exception:
                log.exception("Failed to clean up orphaned upload object %s", key)
            raise
        return saved_doc, key

    async def execute(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        **options: Unpack[UploadOptions],
    ) -> DocumentDTO:
        user_id = options["user_id"]
        user_kind = options["user_kind"]
        user_role = options["user_role"]
        doc_domain = options.get("doc_domain")
        ext = Path(filename).suffix.lower()

        if ext not in self._file_storage.supported_extensions:
            msg = f"Unsupported file format: {ext}"
            raise ValidationError(msg)

        vis = DocumentVisibility.validate(visibility)
        storage_deletes: list[str] = []
        async with self._uow_factory.create(master=True) as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            await self.validate_scope(uow, vis, group_id, ctx)
            effective_owner_id = await self.resolve_effective_owner_id(
                uow,
                vis,
                user_id,
                user_kind,
                client_id=options.get("client_id"),
                user_role=user_role,
                managed_client_ids=ctx.managed_client_ids if user_role == UserRole.CURATOR else None,
            )
            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, effective_owner_id)
            existing = await uow.documents.find_active_slot(
                owner_id, filename, effective_group_id, for_update=True
            )
            self.validate_no_active_processing(existing)
            version_group_id, pending_replace_id, existing = await resolve_upload_version_group(
                uow, existing, options.get("replaces_document_id"), None
            )

            if existing and existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
                filename = await resolve_unique_filename(
                    uow.documents, owner_id, effective_group_id, filename
                )

            doc = Document(
                filename=filename,
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
                doc_domain=doc_domain or DocDomain.GENERAL.value,
                version_group_id=version_group_id,
            )
            saved_doc, key = await self.persist_upload(
                uow, doc, file_data, owner_id, effective_group_id=effective_group_id, filename=filename
            )
            pending_replace_id = await resolve_upload_conflict(
                doc,
                existing,
                pending_replace_id,
                doc_domain,
                uow=uow,
                storage_deletes=storage_deletes,
                domain_registry=self._domain_registry,
            )
            saved_id = self.require_document_id(saved_doc)
            final_doc = await uow.documents.get_by_id(saved_id)

            if final_doc is None:
                msg = "Document"
                raise EntityNotFound(msg, saved_id)

            dto = DocumentDTO.from_entity(final_doc, storage_key=key, replace_id=pending_replace_id)
        await self.cleanup_replaced_objects(storage_deletes)
        return dto

    async def validate_scope(
        self, uow: UnitOfWork, vis: DocumentVisibility, group_id: int | None, ctx: UserContext
    ) -> None:
        validate_document_visibility(vis, group_id, ctx)

        if vis == DocumentVisibility.INTERNAL_GROUP:
            if group_id is None:
                msg = "group_id required for internal_group visibility"
                raise ValidationError(msg)

            groups = await uow.groups.list_by_ids([group_id])

            if not groups:
                msg = "Group"
                raise EntityNotFound(msg, group_id)

    async def cleanup_replaced_objects(self, storage_deletes: list[str]) -> None:
        for old_key in storage_deletes:
            try:
                await self._file_storage.delete_file(old_key)
            except Exception:
                log.exception("Failed to delete replaced document object %s from storage — orphaned", old_key)

    @staticmethod
    def require_document_id(doc: Document) -> int:
        if doc.id is None:
            msg = "Document save returned None id"
            raise RuntimeError(msg)

        return doc.id
