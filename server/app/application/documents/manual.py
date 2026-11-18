"""ManualDocumentService -- creates virtual document containers for manual chunks."""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import TYPE_CHECKING

from application.documents.dto import DocumentDTO
from application.documents.utils import resolve_effective_owner_id
from application.identity import UserContextFactory
from application.spaces.resolution import resolve_space
from domain.knowledge.access_control import compute_owner_and_group, validate_document_visibility
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility, SourceType
from domain.shared.exceptions import PermissionDeniedError


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory
    from domain.identity.roles import UserKind, UserRole

log = logging.getLogger(__name__)

FILL_SPACE_PERMISSION = "knowledge_space.fill"


class ManualDocumentService:
    """Creates virtual document containers for manual chunks."""

    def __init__(
        self, uow_factory: UnitOfWorkFactory, user_ctx_factory: UserContextFactory | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def create_manual_document(  # noqa: PLR0913 - explicit space and audience selectors
        self,
        title: str,
        visibility: DocumentVisibility,
        user_id: int,
        user_kind: UserKind,
        *,
        user_role: UserRole,
        group_id: int | None = None,
        space_id: int | None = None,
        client_id: int | None = None,
    ) -> DocumentDTO:
        vis = DocumentVisibility.parse(visibility)

        async with self._uow_factory.create(master=True) as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            space = await resolve_space(uow, space_id)

            # A manual document fills its space like an upload does.
            if not space.can_be_filled_by(ctx):
                raise PermissionDeniedError(FILL_SPACE_PERMISSION)

            validate_document_visibility(vis, group_id, ctx)

            effective_owner_id = user_id

            if client_id is not None:
                effective_owner_id = await resolve_effective_owner_id(
                    uow,
                    vis,
                    user_id,
                    user_kind,
                    client_id=client_id,
                    user_role=user_role,
                    managed_client_ids=ctx.managed_client_ids,
                )

            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, effective_owner_id)

            doc = Document(
                filename=title,
                source_path="",
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
                space_id=space.id,
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

            return replace(
                DocumentDTO.from_entity(saved_doc), chunks=0, chars=0, source_type=SourceType.MANUAL.value
            )
