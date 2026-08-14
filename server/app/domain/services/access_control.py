"""Access control domain service -- pure business rules for document visibility.

Single source of truth for all visibility/ACL logic.  Both the Qdrant
filter builder and the SQLAlchemy query builder derive their conditions
from ``get_visibility_conditions()`` to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.exceptions import BusinessRuleViolation, ValidationError
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility

# Business rules: which visibility values each user kind can use
ALLOWED_VISIBILITY_FOR_KIND: dict[UserKind, set[DocumentVisibility]] = {
    UserKind.INTERNAL: {
        DocumentVisibility.INTERNAL_PUBLIC,
        DocumentVisibility.INTERNAL_GROUP,
        DocumentVisibility.INTERNAL_PRIVATE,
        DocumentVisibility.CLIENT_PRIVATE,
    },
    UserKind.CLIENT: {DocumentVisibility.CLIENT_PRIVATE},
}


@dataclass(frozen=True)
class VisibilityCondition:
    """A single AND-clause of a visibility filter.

    The full visible-to-user filter is OR of all returned conditions.
    This is the canonical intermediate representation — SQL and Qdrant
    adapters translate these into their respective query languages.
    """

    visibility: DocumentVisibility
    owner_match: str | None = None  # "self" = user_id, "assigned" = assigned_client_ids
    group_match: bool = False  # True = group_id IN user_group_ids


def get_visibility_conditions(
    user_kind: UserKind,
    user_id: int,
    group_ids: list[int],
    assigned_client_ids: list[int],
    for_list: bool = True,
    user_role: UserRole | None = None,
) -> list[VisibilityCondition]:
    """Return canonical filter conditions for documents visible to this user.

    Each condition is an AND-clause. The full filter is OR of all conditions.
    This is the single source of truth — SQL and Qdrant adapters translate these.

    Args:
        for_list: True for document list (admin sees all client docs),
                  False for RAG queries (admin should not search client docs).
        user_role: Required for list mode to distinguish admin from regular users.
    """
    if user_kind == UserKind.CLIENT:
        return [
            VisibilityCondition(
                visibility=DocumentVisibility.CLIENT_PRIVATE,
                owner_match="self",
            )
        ]

    conditions: list[VisibilityCondition] = []
    allowed = ALLOWED_VISIBILITY_FOR_KIND.get(user_kind, set())

    if DocumentVisibility.INTERNAL_PUBLIC in allowed:
        conditions.append(VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC))

    if DocumentVisibility.INTERNAL_PRIVATE in allowed:
        conditions.append(
            VisibilityCondition(
                visibility=DocumentVisibility.INTERNAL_PRIVATE,
                owner_match="self",
            )
        )

    if DocumentVisibility.INTERNAL_GROUP in allowed and group_ids:
        conditions.append(
            VisibilityCondition(
                visibility=DocumentVisibility.INTERNAL_GROUP,
                group_match=True,
            )
        )

    # Admin can view ALL client_private docs in list mode (not search mode)
    if for_list and user_role == UserRole.ADMIN:
        conditions.append(
            VisibilityCondition(
                visibility=DocumentVisibility.CLIENT_PRIVATE,
            )
        )

    return conditions


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

    if (
        visibility == DocumentVisibility.CLIENT_PRIVATE
        and user_kind == UserKind.INTERNAL
        and user_role != UserRole.ADMIN
    ):
        raise BusinessRuleViolation("Only admin can upload documents for clients")

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
    user_role: str | None = None,
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
        # Admin can view any client_private document
        if user_role == UserRole.ADMIN:
            return True
        # Regular internal user can view if they own the doc
        return doc_owner_id == user_id

    return False
