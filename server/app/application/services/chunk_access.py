"""Chunk access helpers -- shared access-check preambles for chunk operations."""

from __future__ import annotations

from typing import Any

from application.services.user_context_factory import UserContextFactory
from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, EntityNotFound
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.user_context import UserContext

_default_factory = UserContextFactory()


async def _load_doc_and_check_permission(
    uow: Any,
    document_id: int,
    user_id: int,
    user_role: str,
    *,
    user_ctx_factory: UserContextFactory | None = None,
) -> tuple[Document, UserContext]:
    doc = await uow.documents.get_by_id(document_id)
    if doc is None:
        raise EntityNotFound("Document", document_id)

    role = UserRole(user_role)
    factory = user_ctx_factory or _default_factory
    ctx = await factory.build(uow, user_id, UserKind.INTERNAL, user_role)

    if not doc.can_edit_chunks(
        user_id,
        role,
        ctx.group_ids,
        managed_client_ids=ctx.managed_client_ids,
        managed_internal_ids=ctx.managed_internal_ids,
        managed_group_ids=ctx.managed_group_ids,
    ):
        raise BusinessRuleViolation("No permission to edit chunks for this document")

    return doc, ctx


async def load_doc_for_edit(
    uow: Any,
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
    return await _load_doc_and_check_permission(
        uow, document_id, user_id, user_role, user_ctx_factory=user_ctx_factory
    )


async def load_doc_for_add(
    uow: Any,
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
        raise EntityNotFound("Document", document_id)

    if doc.status not in (DocumentStatus.DONE, DocumentStatus.INDEXING):
        raise BusinessRuleViolation(
            "Can only add chunks to documents with status 'done' or 'indexing'"
        )

    role = UserRole(user_role)
    factory = user_ctx_factory or _default_factory
    ctx = await factory.build(uow, user_id, UserKind.INTERNAL, user_role)

    if not doc.can_edit_chunks(
        user_id,
        role,
        ctx.group_ids,
        managed_client_ids=ctx.managed_client_ids,
        managed_internal_ids=ctx.managed_internal_ids,
        managed_group_ids=ctx.managed_group_ids,
    ):
        raise BusinessRuleViolation("No permission to add chunks for this document")

    return doc, ctx
