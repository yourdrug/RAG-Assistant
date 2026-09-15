"""ManualDocumentService -- creates virtual document containers for manual chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.dto.document_dto import DocumentDTO
from application.services.user_context_factory import UserContextFactory
from domain.entities.document import Document
from domain.services import compute_owner_and_group, validate_document_visibility
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.source_type import SourceType
from domain.value_objects.visibility import DocumentVisibility

if TYPE_CHECKING:
    from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger(__name__)


class ManualDocumentService:
    """Creates virtual document containers for manual chunks."""

    def __init__(
        self, uow_factory: "UnitOfWorkFactory", user_ctx_factory: UserContextFactory | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def create_manual_document(
        self,
        title: str,
        visibility: str,
        user_id: int,
        user_kind: str,
        user_role: str,
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
