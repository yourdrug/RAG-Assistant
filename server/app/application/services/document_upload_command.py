"""Upload command: validate scope, reserve a name, store and resolve replacement."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from application.dto.document_dto import DocumentDTO
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_conflict_resolver import (
    resolve_upload_conflict,
    resolve_upload_version_group,
)
from application.services.document_utils import generate_storage_key, resolve_unique_filename
from application.services.user_context_factory import UserContextFactory
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
    from domain.domain_profile.registry import DomainProfileRegistry

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

    async def _resolve_effective_owner_id(
        self,
        uow,
        vis,
        user_id,
        user_kind,
        client_id,
        user_role=UserRole.USER,
        managed_client_ids=None,
    ) -> int:
        if vis != DocumentVisibility.CLIENT_PRIVATE:
            return user_id
        if user_kind == UserKind.CLIENT:
            return user_id
        if client_id is None:
            raise ValidationError("client_id required for client_private upload")
        client_user = await uow.users.get_by_id(client_id)
        if client_user is None or client_user.kind != UserKind.CLIENT:
            raise ValidationError("client_id must be a user with kind='client'")
        if user_role == UserRole.CURATOR:
            if managed_client_ids is None or client_id not in managed_client_ids:
                raise BusinessRuleViolation("You can only upload documents for assigned clients")
        return client_id

    @staticmethod
    def _validate_no_active_processing(existing) -> None:
        if existing and existing.status in (
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
            DocumentStatus.INDEXING,
        ):
            raise BusinessRuleViolation("This document is already being processed")

    async def _persist_upload(self, uow, doc, file_data, owner_id, effective_group_id, filename):
        try:
            saved_doc = await uow.documents.save(doc)
        except UniqueConstraintViolation as exc:
            raise BusinessRuleViolation(
                "This document is already being uploaded by a concurrent request"
            ) from exc
        if saved_doc.id is None:
            raise RuntimeError("Document save returned None id")
        key = generate_storage_key(owner_id, effective_group_id, saved_doc.id, filename)
        try:
            await self._file_storage.upload_file(key, file_data)
            await uow.documents.set_source_path(saved_doc.id, key)
        except BaseException:
            try:
                await self._file_storage.delete_file(key)
            except Exception:
                log.warning("Failed to clean up orphaned upload object %s", key)
            raise
        return saved_doc, key

    async def execute(
        self,
        filename,
        file_data,
        visibility,
        group_id,
        user_id,
        user_kind,
        user_role,
        client_id=None,
        rename_on_conflict=False,
        doc_domain=None,
        replaces_document_id=None,
    ) -> DocumentDTO:
        ext = Path(filename).suffix.lower()
        if ext not in self._file_storage.supported_extensions:
            raise ValidationError(f"Unsupported file format: {ext}")
        vis = DocumentVisibility.validate(visibility)
        storage_deletes: list[str] = []
        async with self._uow_factory.create(master=True) as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            validate_document_visibility(vis, group_id, ctx)
            if vis == DocumentVisibility.INTERNAL_GROUP:
                if group_id is None:
                    raise ValidationError("group_id required for internal_group visibility")
                groups = await uow.groups.list_by_ids([group_id])
                if not groups:
                    raise EntityNotFound("Group", group_id)
            effective_owner_id = await self._resolve_effective_owner_id(
                uow,
                vis,
                user_id,
                user_kind,
                client_id,
                user_role=user_role,
                managed_client_ids=ctx.managed_client_ids if user_role == UserRole.CURATOR else None,
            )
            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, effective_owner_id)
            existing = await uow.documents.find_active_slot(
                owner_id, filename, effective_group_id, for_update=True
            )
            self._validate_no_active_processing(existing)
            version_group_id, pending_replace_id, existing = await resolve_upload_version_group(
                uow, existing, replaces_document_id, None
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
            saved_doc, key = await self._persist_upload(
                uow, doc, file_data, owner_id, effective_group_id, filename
            )
            pending_replace_id = await resolve_upload_conflict(
                doc,
                existing,
                pending_replace_id,
                doc_domain,
                uow,
                storage_deletes,
                domain_registry=self._domain_registry,
            )
            final_doc = await uow.documents.get_by_id(saved_doc.id)
            if final_doc is None:
                raise EntityNotFound("Document", saved_doc.id)
            dto = DocumentDTO.from_entity(final_doc, storage_key=key, replace_id=pending_replace_id)
        for old_key in storage_deletes:
            try:
                await self._file_storage.delete_file(old_key)
            except Exception:
                log.warning("Failed to delete replaced document object %s from storage — orphaned", old_key)
        return dto
