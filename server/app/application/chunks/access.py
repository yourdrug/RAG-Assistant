"""Chunk access helpers -- shared access-check preambles for chunk operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.identity import UserContextFactory
from domain.exceptions import BusinessRuleViolation, EntityNotFound
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole


if TYPE_CHECKING:
    from application.uow import UnitOfWork
    from domain.entities.document import Document
    from domain.value_objects.user_context import UserContext


_default_factory = UserContextFactory()


async def load_doc_and_check_permission(
    uow: UnitOfWork,
    document_id: int,
    user_id: int,
    user_role: str,
    *,
    user_ctx_factory: UserContextFactory | None = None,
) -> tuple[Document, UserContext]:
    doc = await uow.documents.get_by_id(document_id)

    if doc is None:
        msg = "Document"
        raise EntityNotFound(msg, document_id)

    role = UserRole(user_role)
    factory = user_ctx_factory or _default_factory
    ctx = await factory.build(uow, user_id, UserKind.INTERNAL, user_role)

    if not doc.can_edit_chunks(
        user_id,
        role,
        list(ctx.group_ids),
        managed_client_ids=list(ctx.managed_client_ids),
        managed_internal_ids=list(ctx.managed_internal_ids),
        managed_group_ids=list(ctx.managed_group_ids),
    ):
        msg = "No permission to edit chunks for this document"
        raise BusinessRuleViolation(msg)

    return doc, ctx


async def load_doc_for_edit(
    uow: UnitOfWork,
    document_id: int,
    user_id: int,
    user_role: str,
    *,
    user_ctx_factory: UserContextFactory | None = None,
) -> tuple[Document, UserContext]:
    """Load document and verify edit permission.

    Order matches the original ChunkService.edit_chunk: doc load ->
    UserContext -> permission check.
    """
    return await load_doc_and_check_permission(
        uow, document_id, user_id, user_role, user_ctx_factory=user_ctx_factory
    )


async def load_doc_for_add(
    uow: UnitOfWork,
    document_id: int,
    user_id: int,
    user_role: str,
    *,
    user_ctx_factory: UserContextFactory | None = None,
) -> tuple[Document, UserContext]:
    """Load document, verify status first, then edit permission.

    Order matches the original ChunkService.add_chunk: doc load ->
    status check -> UserContext -> permission check.
    """
    doc = await uow.documents.get_by_id(document_id)

    if doc is None:
        msg = "Document"
        raise EntityNotFound(msg, document_id)

    if doc.status not in (DocumentStatus.DONE, DocumentStatus.INDEXING):
        msg = "Can only add chunks to documents with status 'done' or 'indexing'"
        raise BusinessRuleViolation(msg)

    role = UserRole(user_role)
    factory = user_ctx_factory or _default_factory
    ctx = await factory.build(uow, user_id, UserKind.INTERNAL, user_role)

    if not doc.can_edit_chunks(
        user_id,
        role,
        list(ctx.group_ids),
        managed_client_ids=list(ctx.managed_client_ids),
        managed_internal_ids=list(ctx.managed_internal_ids),
        managed_group_ids=list(ctx.managed_group_ids),
    ):
        msg = "No permission to add chunks for this document"
        raise BusinessRuleViolation(msg)

    return doc, ctx
