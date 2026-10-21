"""Upload command: validate scope, reserve a name and defer replacement to processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

from application.documents.conflicts import resolve_upload_version_group
from application.documents.dto import DocumentDTO
from application.documents.utils import generate_storage_key, resolve_unique_filename
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import (
    check_ownership,
    compute_owner_and_group,
    validate_document_visibility,
)
from domain.knowledge.acts.policy import same_document_scope
from domain.knowledge.document import Document
from domain.knowledge.enums import DocDomain, DocumentStatus, DocumentVisibility
from domain.shared.exceptions import (
    BusinessRuleViolationError,
    EntityNotFoundError,
    UniqueConstraintError,
    ValidationError,
)


if TYPE_CHECKING:
    from application.documents.dto import UploadOptions
    from application.identity import UserContextFactory
    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.identity.user_context import UserContext

log = logging.getLogger("default")


class DocumentUploadCommand:
    """Own the upload transaction and compensation for a failed storage write."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        file_storage: FileStorage,
        user_ctx_factory: UserContextFactory,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._user_ctx_factory = user_ctx_factory

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
            raise BusinessRuleViolationError(msg)

        return client_id

    @staticmethod
    def validate_no_active_processing(existing: Document | None) -> None:
        if existing and existing.status in (
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
            DocumentStatus.INDEXING,
        ):
            msg = "This document is already being processed"
            raise BusinessRuleViolationError(msg)

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
        except UniqueConstraintError as exc:
            msg = "This document is already being uploaded by a concurrent request"
            raise BusinessRuleViolationError(msg) from exc

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
        rename_on_conflict = options.get("rename_on_conflict", False)
        replaces_document_id = options.get("replaces_document_id")

        if rename_on_conflict and replaces_document_id is not None:
            msg = "Cannot replace a document when rename_on_conflict is enabled"
            raise ValidationError(msg)

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
            doc = Document(
                filename=filename,
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
                doc_domain=doc_domain or DocDomain.GENERAL.value,
            )
            pending_replace_id = await self.prepare_upload_target(
                uow,
                doc,
                ctx,
                rename_on_conflict=rename_on_conflict,
                replaces_document_id=replaces_document_id,
            )
            saved_doc, key = await self.persist_upload(
                uow, doc, file_data, owner_id, effective_group_id=effective_group_id, filename=doc.filename
            )
            saved_id = self.require_document_id(saved_doc)
            final_doc = await uow.documents.get_by_id(saved_id)

            if final_doc is None:
                msg = "Document"
                raise EntityNotFoundError(msg, saved_id)

            return DocumentDTO.from_entity(final_doc, storage_key=key, replace_id=pending_replace_id)

    async def prepare_upload_target(
        self,
        uow: UnitOfWork,
        doc: Document,
        ctx: UserContext,
        *,
        rename_on_conflict: bool,
        replaces_document_id: int | None,
    ) -> int | None:
        existing = await uow.documents.find_active_slot(
            doc.owner_id, doc.filename, doc.group_id, for_update=True
        )
        pending_replace_id = None

        if not rename_on_conflict:
            self.validate_no_active_processing(existing)
            doc.version_group_id, pending_replace_id, existing = await resolve_upload_version_group(
                uow, existing, replaces_document_id, None
            )
            self.validate_no_active_processing(existing)

            if existing is not None:
                check_ownership(existing, ctx, "replace")

                if not same_document_scope(existing, doc):
                    msg = "Replacement must have the same visibility scope"
                    raise BusinessRuleViolationError(msg)

        if existing:
            doc.filename = await resolve_unique_filename(
                uow.documents, doc.owner_id, doc.group_id, doc.filename
            )

        return pending_replace_id

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
                raise EntityNotFoundError(msg, group_id)

    @staticmethod
    def require_document_id(doc: Document) -> int:
        if doc.id is None:
            msg = "Document save returned None id"
            raise RuntimeError(msg)

        return doc.id
