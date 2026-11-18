"""Upload command: validate scope, reserve a name and defer replacement to processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

from application.documents.conflicts import resolve_upload_version_group
from application.documents.dto import DocumentDTO
from application.documents.utils import (
    generate_storage_key,
    resolve_effective_owner_id,
    resolve_unique_filename,
)
from application.ingestion.profiles import ensure_known_domain
from application.spaces.resolution import resolve_space
from domain.identity.roles import UserRole
from domain.knowledge.access_control import (
    check_ownership,
    compute_owner_and_group,
    validate_document_visibility,
)
from domain.knowledge.document import Document
from domain.knowledge.document_scope import same_document_scope
from domain.knowledge.enums import DocDomain, DocumentStatus, DocumentVisibility, DomainKey
from domain.runtime.jobs import BackgroundJob, JobType
from domain.shared.exceptions import (
    BusinessRuleViolationError,
    EntityNotFoundError,
    PermissionDeniedError,
    UniqueConstraintError,
    ValidationError,
)


if TYPE_CHECKING:
    from application.documents.dto import UploadOptions
    from application.identity import UserContextFactory
    from application.ports.file_storage import FileStorage
    from application.spaces.ports import DomainPackCatalog
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.identity.user_context import UserContext
    from domain.knowledge.space import KnowledgeSpace

log = logging.getLogger("default")

FILL_SPACE_PERMISSION = "knowledge_space.fill"


class DocumentUploadCommand:
    """Own the upload transaction and compensation for a failed storage write."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        file_storage: FileStorage,
        user_ctx_factory: UserContextFactory,
        pack_catalog: DomainPackCatalog,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._user_ctx_factory = user_ctx_factory
        self._pack_catalog = pack_catalog

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

        vis = DocumentVisibility.parse(visibility)
        rename_on_conflict = options.get("rename_on_conflict", False)
        replaces_document_id = options.get("replaces_document_id")

        if rename_on_conflict and replaces_document_id is not None:
            msg = "Cannot replace a document when rename_on_conflict is enabled"
            raise ValidationError(msg)

        async with self._uow_factory.create(master=True) as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            space = await self.resolve_target_space(uow, options.get("space_id"), ctx, doc_domain)
            await self.validate_scope(uow, vis, group_id, ctx)
            effective_owner_id = await resolve_effective_owner_id(
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
                space_id=space.id,
                doc_domain=DomainKey(doc_domain or DocDomain.GENERAL),
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

            # This job row is also the task outbox: commit it with the document.
            # Redis delivery happens later, without depending on the HTTP process.
            await self.record_processing_job(
                uow,
                BackgroundJob(
                    job_type=JobType.DOCUMENT_PROCESSING,
                    related_id=saved_id,
                    dispatch_payload={
                        "document_id": saved_id,
                        "storage_key": key,
                        "filename": final_doc.filename,
                        "visibility": final_doc.visibility.value,
                        "owner_id": final_doc.owner_id,
                        "group_id": final_doc.group_id,
                        "replace_id": pending_replace_id,
                        "doc_domain": doc_domain,
                        "principal_id": user_id,
                        "cost_principal": f"user:{user_id}",
                    },
                ),
                key=key,
            )
            return DocumentDTO.from_entity(final_doc, storage_key=key, replace_id=pending_replace_id)

    async def record_processing_job(self, uow: UnitOfWork, job: BackgroundJob, *, key: str) -> None:
        try:
            await uow.background_jobs.create(job)
        except BaseException:
            try:
                await self._file_storage.delete_file(key)
            except Exception:
                log.exception("Failed to compensate a rejected upload")
            raise

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
            doc.owner_id, doc.filename, doc.group_id, space_id=doc.space_id, for_update=True
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
                    msg = "Replacement must have the same knowledge space and visibility scope"
                    raise BusinessRuleViolationError(msg)

        if existing:
            doc.filename = await resolve_unique_filename(
                uow.documents, doc.owner_id, doc.group_id, doc.filename, space_id=doc.space_id
            )

        return pending_replace_id

    async def resolve_target_space(
        self, uow: UnitOfWork, space_id: int | None, ctx: UserContext, doc_domain: str | None
    ) -> KnowledgeSpace:
        """The space being filled: only admitted groups may add or replace its documents."""
        space = await resolve_space(uow, space_id)

        if not space.can_be_filled_by(ctx):
            raise PermissionDeniedError(FILL_SPACE_PERMISSION)

        if doc_domain is not None:
            # Classification runs among the profiles of the space's pack; so does an explicit choice.
            ensure_known_domain(
                self._pack_catalog.domain_registry(space.pack_key), doc_domain, field="doc_domain"
            )

        return space

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
