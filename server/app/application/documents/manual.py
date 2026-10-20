"""ManualDocumentService -- creates virtual document containers for manual chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.documents.dto import DocumentDTO
from application.identity import UserContextFactory
from domain.knowledge.access_control import compute_owner_and_group, validate_document_visibility
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility, SourceType


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory
    from domain.identity.roles import UserKind, UserRole

log = logging.getLogger(__name__)


class ManualDocumentService:
    """Creates virtual document containers for manual chunks."""

    def __init__(
        self, uow_factory: UnitOfWorkFactory, user_ctx_factory: UserContextFactory | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def create_manual_document(
        self,
        title: str,
        visibility: DocumentVisibility,
        user_id: int,
        user_kind: UserKind,
        *,
        user_role: UserRole,
        group_id: int | None = None,
    ) -> DocumentDTO:
        vis = DocumentVisibility.validate(visibility)

        async with self._uow_factory.create(master=True) as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            validate_document_visibility(vis, group_id, ctx)

            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, user_id)

            doc = Document(
                filename=title,
                source_path="",
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
                status=DocumentStatus.DONE,
                source_type=SourceType.MANUAL.value,
                chunks=0,
                chars=0,
            )

            saved_doc = await uow.documents.save(doc)

            log.info(
                "Manual document %d created by user %d: %s",
                saved_doc.id,
                user_id,
                title,
            )

            return DocumentDTO.from_entity(saved_doc, chunks=0, chars=0, source_type=SourceType.MANUAL.value)
