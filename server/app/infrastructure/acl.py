"""
infrastructure/acl.py — Qdrant filter construction for document access control.

Translates canonical VisibilityCondition objects from the domain layer
into Qdrant Filter objects. All business logic lives in domain/services/access_control.py.
"""

from domain.services.access_control import get_visibility_conditions
from domain.value_objects.roles import UserKind
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


def build_qdrant_filter(
    user: dict,
    group_ids: list[int],
    assigned_client_ids: list[int],
) -> Filter:
    """Build a Qdrant filter for documents visible to this user.

    Derives conditions from domain.services.access_control.get_visibility_conditions().

    Args:
        user: dict with "id" and "kind" keys
        group_ids: pre-fetched group IDs for this user
        assigned_client_ids: pre-fetched assigned client IDs for this user
    """
    conditions = get_visibility_conditions(UserKind(user["kind"]), user["id"], group_ids, assigned_client_ids)

    should: list[FieldCondition | Filter] = []

    for cond in conditions:
        must = [FieldCondition(key="visibility", match=MatchValue(value=cond.visibility))]

        if cond.owner_match == "self":
            must.append(FieldCondition(key="owner_id", match=MatchValue(value=user["id"])))
        elif cond.owner_match == "assigned":
            must.append(FieldCondition(key="owner_id", match=MatchAny(any=assigned_client_ids)))

        if cond.group_match:
            must.append(FieldCondition(key="group_id", match=MatchAny(any=group_ids)))

        should.append(Filter(must=must) if len(must) > 1 else must[0])

    return Filter(should=should)
