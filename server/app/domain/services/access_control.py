"""Access Control Domain Service — pure business rules for document visibility.

This is the domain logic previously in infrastructure/acl.py, now properly
in the domain layer with no infrastructure dependencies.
"""

from __future__ import annotations

from domain.exceptions import BusinessRuleViolation, ValidationError
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility

# Business rules: which visibility values each user kind can use
ALLOWED_VISIBILITY_FOR_KIND: dict[UserKind, set[DocumentVisibility]] = {
    UserKind.INTERNAL: {
        DocumentVisibility.INTERNAL_PUBLIC,
        DocumentVisibility.INTERNAL_GROUP,
        DocumentVisibility.INTERNAL_PRIVATE,
    },
    UserKind.CLIENT: {DocumentVisibility.CLIENT_PRIVATE},
}


def validate_document_visibility(
    visibility: DocumentVisibility,
    group_id: int | None,
    user_kind: UserKind,
    user_role: UserRole,
    user_group_ids: list[int],
) -> None:
    """Business rule: validate that a user can use the given visibility."""
    allowed = ALLOWED_VISIBILITY_FOR_KIND.get(user_kind)
    if allowed is None or visibility not in allowed:
        raise ValidationError(f"visibility='{visibility}' not available for kind='{user_kind}'")

    if visibility == DocumentVisibility.INTERNAL_PUBLIC and user_role != UserRole.ADMIN:
        raise BusinessRuleViolation("Only admin can publish to internal_public")

    if visibility == DocumentVisibility.INTERNAL_GROUP:
        if group_id is None:
            raise ValidationError("group_id required for visibility='internal_group'")
        if group_id not in user_group_ids:
            raise BusinessRuleViolation("You are not a member of this group")


def compute_owner_and_group(
    visibility: DocumentVisibility,
    group_id: int | None,
    user_id: int,
) -> tuple[int | None, int | None]:
    """Business rule: determine owner_id and group_id for a document based on visibility."""
    if visibility == DocumentVisibility.INTERNAL_PUBLIC:
        return None, None
    if visibility == DocumentVisibility.INTERNAL_GROUP:
        return None, group_id
    return user_id, None


def can_view_document(
    doc_visibility: str,
    doc_owner_id: int | None,
    doc_group_id: int | None,
    user_kind: str,
    user_id: int,
    user_group_ids: list[int],
    assigned_client_ids: list[int],
) -> bool:
    """Business rule: can this user view this document?"""
    vis = DocumentVisibility(doc_visibility)
    kind = UserKind(user_kind)

    if vis == DocumentVisibility.INTERNAL_PUBLIC:
        return kind == UserKind.INTERNAL

    if vis == DocumentVisibility.INTERNAL_GROUP:
        return kind == UserKind.INTERNAL and doc_group_id in user_group_ids

    if vis == DocumentVisibility.INTERNAL_PRIVATE:
        return kind == UserKind.INTERNAL and doc_owner_id == user_id

    if vis == DocumentVisibility.CLIENT_PRIVATE:
        if kind == UserKind.CLIENT:
            return doc_owner_id == user_id
        return doc_owner_id in assigned_client_ids

    return False
