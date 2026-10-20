"""Chunk access helpers -- shared access-check preambles for chunk operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.identity import UserContextFactory
from domain.identity.roles import UserKind
from domain.knowledge.enums import DocumentStatus
from domain.shared.exceptions import BusinessRuleViolation, EntityNotFound


if TYPE_CHECKING:
    from application.uow import UnitOfWork
    from domain.identity.roles import UserRole
    from domain.identity.user_context import UserContext
    from domain.knowledge.document import Document


_default_factory = UserContextFactory()


async def load_doc_for_edit(
    uow: UnitOfWork,
    document_id: int,
    user_id: int,
    user_role: UserRole,
    *,
    user_ctx_factory: UserContextFactory | None = None,
) -> tuple[Document, UserContext]:
    doc = await uow.documents.get_by_id(document_id)

    if doc is None:
        msg = "Document"
        raise EntityNotFound(msg, document_id)

    factory = user_ctx_factory or _default_factory
    ctx = await factory.build(uow, user_id, UserKind.INTERNAL, user_role)

    if not doc.can_edit_chunks(
        user_id,
        user_role,
        list(ctx.group_ids),
        managed_client_ids=list(ctx.managed_client_ids),
        managed_internal_ids=list(ctx.managed_internal_ids),
        managed_group_ids=list(ctx.managed_group_ids),
    ):
        msg = "No permission to edit chunks for this document"
        raise BusinessRuleViolation(msg)

    return doc, ctx


async def load_doc_for_add(
    uow: UnitOfWork,
    document_id: int,
    user_id: int,
    user_role: UserRole,
    *,
    user_ctx_factory: UserContextFactory | None = None,
) -> tuple[Document, UserContext]:
    """Load document, verify status first, then edit permission.

    Keep the order: document load, status check, user context, permission check.
    """
    # Serialize index allocation, including documents without an ActVersion.
    doc = await uow.documents.get_by_id(document_id, for_update=True)

    if doc is None:
        msg = "Document"
        raise EntityNotFound(msg, document_id)

    if doc.status not in (DocumentStatus.DONE, DocumentStatus.INDEXING):
        msg = "Can only add chunks to documents with status 'done' or 'indexing'"
        raise BusinessRuleViolation(msg)

    factory = user_ctx_factory or _default_factory
    ctx = await factory.build(uow, user_id, UserKind.INTERNAL, user_role)

    if not doc.can_edit_chunks(
        user_id,
        user_role,
        list(ctx.group_ids),
        managed_client_ids=list(ctx.managed_client_ids),
        managed_internal_ids=list(ctx.managed_internal_ids),
        managed_group_ids=list(ctx.managed_group_ids),
    ):
        msg = "No permission to add chunks for this document"
        raise BusinessRuleViolation(msg)

    return doc, ctx
