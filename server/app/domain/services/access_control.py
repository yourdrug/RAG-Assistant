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
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility

if False:  # TYPE_CHECKING
    from domain.entities.document import Document

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
    This is the canonical intermediate representation -- SQL and Qdrant
    adapters translate these into their respective query languages.
    """

    visibility: DocumentVisibility
    owner_match: str | None = None  # OwnerMatch.SELF = user_id
    owner_ids: list[int] | None = None  # OwnerMatch.ASSIGNED = owner_id in this list
    group_match: bool = False  # True = group_id IN user_group_ids
    group_ids: list[int] | None = None  # explicit group id list for group_match


def get_visibility_conditions(  # noqa: C901
    user_kind: UserKind,
    user_id: int,
    group_ids: list[int],
    for_list: bool = True,
    user_role: UserRole | None = None,
    managed_client_ids: list[int] | None = None,
    managed_internal_ids: list[int] | None = None,
    managed_group_ids: list[int] | None = None,
) -> list[VisibilityCondition]:
    """Return canonical filter conditions for documents visible to this user.

    Each condition is an AND-clause. The full filter is OR of all conditions.
    This is the single source of truth -- SQL and Qdrant adapters translate these.

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

    # --- CURATOR branch: managed users and groups ---
    if user_role == UserRole.CURATOR:
        _managed_client_ids = managed_client_ids or []
        _managed_internal_ids = managed_internal_ids or []
        _managed_group_ids = managed_group_ids or []

        # CURATOR sees assigned internal_private docs
        if _managed_internal_ids:
            conditions.append(
                VisibilityCondition(
                    visibility=DocumentVisibility.INTERNAL_PRIVATE,
                    owner_match=OwnerMatch.ASSIGNED,
                    owner_ids=_managed_internal_ids,
                )
            )

        # CURATOR sees assigned client_private docs
        if _managed_client_ids:
            conditions.append(
                VisibilityCondition(
                    visibility=DocumentVisibility.CLIENT_PRIVATE,
                    owner_match=OwnerMatch.ASSIGNED,
                    owner_ids=_managed_client_ids,
                )
            )

        # CURATOR sees assigned group docs (union of own groups + managed groups)
        all_group_ids = list(set(group_ids) | set(_managed_group_ids))
        if all_group_ids:
            conditions.append(
                VisibilityCondition(
                    visibility=DocumentVisibility.INTERNAL_GROUP,
                    group_match=True,
                    group_ids=all_group_ids,
                )
            )

        # CURATOR: for_list == for_search (no admin bonus)
        return conditions

    # --- ADMIN branch ---
    # Admin can view ALL internal_group docs regardless of group membership
    if user_role == UserRole.ADMIN:
        conditions.append(
            VisibilityCondition(
                visibility=DocumentVisibility.INTERNAL_GROUP,
            )
        )

    # Admin can view ALL internal_private docs regardless of ownership
    if user_role == UserRole.ADMIN:
        conditions.append(
            VisibilityCondition(
                visibility=DocumentVisibility.INTERNAL_PRIVATE,
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
    ctx: UserContext,
) -> None:
    """Validate that a user can use the given visibility."""
    user_kind = UserKind(ctx.user_kind)
    user_role = UserRole(ctx.user_role)
    allowed = ALLOWED_VISIBILITY_FOR_KIND.get(user_kind)
    if allowed is None or visibility not in allowed:
        raise ValidationError(f"visibility='{visibility}' not available for kind='{user_kind}'")

    if visibility == DocumentVisibility.INTERNAL_PUBLIC and user_role != UserRole.ADMIN:
        raise BusinessRuleViolation("Only admin can publish to internal_public")

    if visibility == DocumentVisibility.CLIENT_PRIVATE and user_kind == UserKind.INTERNAL:
        if user_role == UserRole.ADMIN:
            pass  # admin always allowed
        elif user_role == UserRole.CURATOR:
            # curator can upload client_private only for managed clients
            # actual client_id validation happens in DocumentService
            pass
        else:
            raise BusinessRuleViolation("Only admin can upload documents for clients")

    if visibility == DocumentVisibility.INTERNAL_GROUP:
        if group_id is None:
            raise ValidationError("group_id required for visibility='internal_group'")
        all_group_ids = set(ctx.group_ids) | set(ctx.managed_group_ids or [])
        if user_role != UserRole.ADMIN and group_id not in all_group_ids:
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


def can_view_document(doc: Document, ctx: UserContext) -> bool:
    """Determine if the user can view the document.

    Uses ``get_visibility_conditions(for_list=True)`` -- the same canonical
    source of truth used by ``is_in_search_scope`` (for_list=False) and
    ``build_qdrant_filter``.
    """
    DocumentVisibility.validate(doc.visibility)
    conditions = get_visibility_conditions(
        UserKind(ctx.user_kind),
        ctx.user_id,
        ctx.group_ids,
        for_list=True,
        user_role=UserRole(ctx.user_role),
        managed_client_ids=ctx.managed_client_ids,
        managed_internal_ids=ctx.managed_internal_ids,
        managed_group_ids=ctx.managed_group_ids,
    )
    return _matches_any_condition(doc, ctx, conditions)


def is_in_search_scope(doc: Document, ctx: UserContext) -> bool:
    """Check if a document participates in the user's RAG search.

    Uses ``get_visibility_conditions(for_list=False)`` -- the same filter
    that ``build_qdrant_filter`` applies.  Admin does NOT get the
    ``CLIENT_PRIVATE`` bonus in search mode, so cross-client private docs
    are excluded from their search scope.  CURATOR sees the same set in
    both list and search modes.
    """
    conditions = get_visibility_conditions(
        UserKind(ctx.user_kind),
        ctx.user_id,
        ctx.group_ids,
        for_list=False,
        user_role=UserRole(ctx.user_role),
        managed_client_ids=ctx.managed_client_ids,
        managed_internal_ids=ctx.managed_internal_ids,
        managed_group_ids=ctx.managed_group_ids,
    )
    return _matches_any_condition(doc, ctx, conditions)


def _matches_any_condition(doc: Document, ctx: UserContext, conditions: list[VisibilityCondition]) -> bool:
    """Check if a document matches any of the given visibility conditions."""
    for cond in conditions:
        if cond.visibility.value != doc.visibility:
            continue
        if cond.owner_match == OwnerMatch.SELF and doc.owner_id != ctx.user_id:
            continue
        if cond.owner_match == OwnerMatch.ASSIGNED:
            if doc.owner_id is None or cond.owner_ids is None or doc.owner_id not in cond.owner_ids:
                continue
        if cond.group_match:
            effective_group_ids = cond.group_ids if cond.group_ids is not None else ctx.group_ids
            if doc.group_id is None or doc.group_id not in effective_group_ids:
                continue
        return True
    return False


def check_document_access(doc: Document, ctx: UserContext) -> None:
    """Raise BusinessRuleViolation if user cannot view the document."""
    if not can_view_document(doc, ctx):
        raise BusinessRuleViolation("No access to this document")


def check_ownership(doc: Document, ctx: UserContext, action: str = "modify") -> None:
    """Raise BusinessRuleViolation if user cannot modify/delete the document."""
    if not doc.can_be_deleted_by(
        ctx.user_id,
        UserRole(ctx.user_role),
        ctx.group_ids,
        managed_client_ids=ctx.managed_client_ids,
        managed_internal_ids=ctx.managed_internal_ids,
        managed_group_ids=ctx.managed_group_ids,
    ):
        raise BusinessRuleViolation(f"Can only {action} your own documents")
