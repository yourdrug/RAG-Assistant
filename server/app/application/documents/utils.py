"""Document utility functions -- storage key generation and filename resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from domain.identity.roles import UserKind, UserRole
from domain.knowledge.enums import DocumentVisibility
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError


if TYPE_CHECKING:
    from application.uow import UnitOfWork
    from domain.knowledge.repositories.document import DocumentCrudRepository

log = logging.getLogger(__name__)


def generate_storage_key(owner_id: int | None, group_id: int | None, document_id: int, filename: str) -> str:
    """Build the storage key path for a document."""
    safe_name = Path(filename).name

    if owner_id is not None:
        return f"uploads/users/{owner_id}/{document_id}_{safe_name}"

    if group_id is not None:
        return f"uploads/groups/{group_id}/{document_id}_{safe_name}"

    return f"uploads/public/{document_id}_{safe_name}"


async def resolve_unique_filename(
    documents: DocumentCrudRepository,
    owner_id: int | None,
    group_id: int | None,
    filename: str,
    *,
    space_id: int | None = None,
) -> str:
    """Find a unique filename (within the space, when given) by appending (N) suffix if needed."""
    p = Path(filename)
    stem = p.stem
    suffix = p.suffix
    candidate = filename
    counter = 1

    while await documents.find_active_slot(owner_id, candidate, group_id, space_id=space_id) is not None:
        candidate = f"{stem}({counter}){suffix}"
        counter += 1

    if candidate != filename:
        log.info("Renamed conflict: %s -> %s", filename, candidate)

    return candidate


async def resolve_effective_owner_id(
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

    if user_role == UserRole.CURATOR and (managed_client_ids is None or client_id not in managed_client_ids):
        msg = "You can only upload documents for assigned clients"
        raise BusinessRuleViolationError(msg)

    return client_id
