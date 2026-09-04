"""Access control domain service -- pure business rules for document visibility.

Single source of truth for all visibility/ACL logic.  Both the Qdrant
filter builder and the SQLAlchemy query builder derive their conditions
from ``get_visibility_conditions()`` to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.exceptions import BusinessRuleViolation, ValidationError
from domain.value_objects.owner_match import OwnerMatch
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
    owner_match: str | None = None  # OwnerMatch.SELF = user_id
    group_match: bool = False  # True = group_id IN user_group_ids


def get_visibility_conditions(
    user_kind: UserKind,
    user_id: int,
    group_ids: list[int],
    for_list: bool = True,
    user_role: UserRole | None = None,
) -> list[VisibilityCondition]:
    """Return canonical filter conditions for documents visible to this user.

    Each condition is an AND-clause. The full filter is OR of all conditions.
    This is the single source of truth — SQL and Qdrant adapters translate these.

    Args:
        user_kind: The kind of user (internal or client).
        user_id: The ID of the user.
        group_ids: List of group IDs the user belongs to.
        for_list: True for document list (admin sees all client docs),
                  False for RAG queries (admin should not search client docs).
        user_role: Required for list mode to distinguish admin from regular users.

    """
    if user_kind == UserKind.CLIENT:
        return [
            VisibilityCondition(
                visibility=DocumentVisibility.CLIENT_PRIVATE,
                owner_match=OwnerMatch.SELF,
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
                owner_match=OwnerMatch.SELF,
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
    """Validate that a user can use the given visibility."""
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
    """Determine owner_id and group_id for a document based on visibility."""
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
    user_role: str | None = None,
) -> bool:
    """Determine if the user can view the document.

    Uses ``get_visibility_conditions(for_list=True)`` — the same canonical
    source of truth used by ``is_in_search_scope`` (for_list=False) and
    ``build_qdrant_filter``.
    """
    conditions = get_visibility_conditions(
        UserKind(user_kind),
        user_id,
        user_group_ids,
        for_list=True,
        user_role=UserRole(user_role) if user_role else None,
    )
    for cond in conditions:
        if cond.visibility.value != doc_visibility:
            continue
        if cond.owner_match == OwnerMatch.SELF and doc_owner_id != user_id:
            continue
        if cond.group_match and (doc_group_id is None or doc_group_id not in user_group_ids):
            continue
        return True
    return False


def is_in_search_scope(
    doc_visibility: str,
    doc_owner_id: int | None,
    doc_group_id: int | None,
    user_kind: str,
    user_id: int,
    user_group_ids: list[int],
    user_role: str | None = None,
) -> bool:
    """Check if a document participates in the user's RAG search.

    Uses ``get_visibility_conditions(for_list=False)`` — the same filter
    that ``build_qdrant_filter`` applies.  Admin does NOT get the
    ``CLIENT_PRIVATE`` bonus in search mode, so cross-client private docs
    are excluded from their search scope.
    """
    conditions = get_visibility_conditions(
        UserKind(user_kind),
        user_id,
        user_group_ids,
        for_list=False,
        user_role=UserRole(user_role) if user_role else None,
    )
    for cond in conditions:
        if cond.visibility.value != doc_visibility:
            continue
        if cond.owner_match == OwnerMatch.SELF and doc_owner_id != user_id:
            continue
        if cond.group_match and (doc_group_id is None or doc_group_id not in user_group_ids):
            continue
        return True
    return False
