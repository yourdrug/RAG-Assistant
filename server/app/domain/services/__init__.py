"""Domain services -- pure business logic (access control, RAG policy, protocols)."""

from domain.services.access_control import (
    VisibilityCondition,
    can_view_document,
    check_document_access,
    check_ownership,
    compute_owner_and_group,
    get_visibility_conditions,
    is_in_search_scope,
    validate_document_visibility,
)

__all__ = [
    "VisibilityCondition",
    "validate_document_visibility",
    "compute_owner_and_group",
    "can_view_document",
    "is_in_search_scope",
    "check_document_access",
    "check_ownership",
    "get_visibility_conditions",
]
