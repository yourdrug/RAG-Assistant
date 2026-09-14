"""
infrastructure/acl.py — Qdrant filter construction for document access control.

Translates canonical VisibilityCondition objects from the domain layer
into Qdrant Filter objects. All business logic lives in domain/services/access_control.py.
"""

from __future__ import annotations

from datetime import date

from domain.services import get_visibility_conditions
from domain.value_objects.roles import UserKind, UserRole
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    HasIdCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchAny,
    MatchValue,
    NestedCondition,
    PayloadField,
)


def build_qdrant_filter(
    user: dict,
    group_ids: list[int],
    managed_client_ids: list[int] | None = None,
    managed_internal_ids: list[int] | None = None,
    managed_group_ids: list[int] | None = None,
) -> Filter:
    """Build a Qdrant filter for documents visible to this user.

    Derives conditions from domain.services.access_control.get_visibility_conditions().

    Args:
        user: dict with "id", "kind", and optionally "role" keys
        group_ids: pre-fetched group IDs for this user
        managed_client_ids: curator's assigned client user IDs
        managed_internal_ids: curator's assigned internal user IDs
        managed_group_ids: curator's assigned group IDs

    """
    user_role = UserRole(user["role"]) if "role" in user and user["role"] else None
    conditions = get_visibility_conditions(
        UserKind(user["kind"]),
        user["id"],
        group_ids,
        for_list=False,
        user_role=user_role,
        managed_client_ids=managed_client_ids,
        managed_internal_ids=managed_internal_ids,
        managed_group_ids=managed_group_ids,
    )

    ConditionType = (
        FieldCondition | IsEmptyCondition | IsNullCondition | HasIdCondition | NestedCondition | Filter
    )
    should: list[ConditionType] = []

    for cond in conditions:
        must: list[ConditionType] = [
            FieldCondition(key="metadata.visibility", match=MatchValue(value=cond.visibility)),
        ]

        if cond.owner_match == "self":
            must.append(FieldCondition(key="metadata.owner_id", match=MatchValue(value=user["id"])))

        if cond.owner_match == "assigned" and cond.owner_ids:
            must.append(FieldCondition(key="metadata.owner_id", match=MatchAny(any=cond.owner_ids)))

        if cond.group_match:
            effective_group_ids = cond.group_ids if cond.group_ids is not None else group_ids
            must.append(FieldCondition(key="metadata.group_id", match=MatchAny(any=effective_group_ids)))

        should.append(Filter(must=must) if len(must) > 1 else must[0])

    return Filter(should=should)


def with_domain_filter(access_filter: Filter, doc_domain: str) -> Filter:
    """Add a doc_domain condition to an existing ACL filter.

    Returns a new Filter that requires both the original ACL conditions AND
    the domain match.
    """
    domain_condition = FieldCondition(
        key="metadata.doc_domain",
        match=MatchValue(value=doc_domain),
    )
    if access_filter is not None:
        return Filter(must=[access_filter, domain_condition])
    return Filter(must=[domain_condition])


def with_temporal_filter(access_filter: Filter, as_of_date: date | None) -> Filter:
    """Add temporal filtering for versioned documents.

    When as_of_date is None: require is_current=True (current state).
    When as_of_date is set: require effective_from <= date AND effective_to > date,
    but chunks WITHOUT dates (NULL) are NEVER excluded — they pass through.

    Args:
        access_filter: existing ACL filter (may be empty)
        as_of_date: date to filter by, or None for current state

    """
    ConditionType = (
        FieldCondition | IsEmptyCondition | IsNullCondition | HasIdCondition | NestedCondition | Filter
    )

    if as_of_date is None:
        temporal_condition: ConditionType = FieldCondition(
            key="metadata.is_current",
            match=MatchValue(value=True),
        )
    else:
        # Chunks with NULL dates always pass (non-versioned content or
        # dates not yet trusted — see TZ section 9.2/11.2).
        temporal_condition = Filter(
            must=[
                Filter(
                    should=[
                        IsNullCondition(is_null=PayloadField(key="metadata.effective_from")),
                        FieldCondition(
                            key="metadata.effective_from",
                            range=DatetimeRange(lte=as_of_date),
                        ),
                    ]
                ),
                Filter(
                    should=[
                        IsNullCondition(is_null=PayloadField(key="metadata.effective_to")),
                        FieldCondition(
                            key="metadata.effective_to",
                            range=DatetimeRange(gt=as_of_date),
                        ),
                    ]
                ),
            ]
        )

    if access_filter is not None:
        return Filter(must=[access_filter, temporal_condition])
    return Filter(must=[temporal_condition])
