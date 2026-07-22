"""
infrastructure/acl.py — Qdrant filter construction for document access control.

Builds Qdrant Filter objects based on user kind and pre-fetched ACL data.
Uses domain/services/access_control.py as the single source of truth for business rules.
"""

from domain.services.access_control import ALLOWED_VISIBILITY_FOR_KIND
from domain.value_objects.roles import UserKind
from domain.value_objects.visibility import DocumentVisibility
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


def build_qdrant_filter(
    user: dict,
    group_ids: list[int],
    assigned_client_ids: list[int],
) -> Filter:
    """Build a Qdrant filter for documents visible to this user.

    Uses ALLOWED_VISIBILITY_FOR_KIND from domain as the single source of truth.

    Args:
        user: dict with "id" and "kind" keys
        group_ids: pre-fetched group IDs for this user
        assigned_client_ids: pre-fetched assigned client IDs for this user
    """
    kind = UserKind(user["kind"])
    allowed = ALLOWED_VISIBILITY_FOR_KIND.get(kind, set())

    if kind == UserKind.CLIENT:
        return Filter(
            must=[
                FieldCondition(key="visibility", match=MatchValue(value=DocumentVisibility.CLIENT_PRIVATE)),
                FieldCondition(key="owner_id", match=MatchValue(value=user["id"])),
            ]
        )

    should: list[FieldCondition | Filter] = []

    if DocumentVisibility.INTERNAL_PUBLIC in allowed:
        should.append(
            FieldCondition(key="visibility", match=MatchValue(value=DocumentVisibility.INTERNAL_PUBLIC))
        )

    if DocumentVisibility.INTERNAL_PRIVATE in allowed:
        should.append(
            Filter(
                must=[
                    FieldCondition(
                        key="visibility", match=MatchValue(value=DocumentVisibility.INTERNAL_PRIVATE)
                    ),
                    FieldCondition(key="owner_id", match=MatchValue(value=user["id"])),
                ]
            )
        )

    if DocumentVisibility.INTERNAL_GROUP in allowed and group_ids:
        should.append(
            Filter(
                must=[
                    FieldCondition(
                        key="visibility", match=MatchValue(value=DocumentVisibility.INTERNAL_GROUP)
                    ),
                    FieldCondition(key="group_id", match=MatchAny(any=group_ids)),
                ]
            )
        )

    # Internal users can view client_private docs of their assigned clients
    # (not in ALLOWED_VISIBILITY_FOR_KIND because they can't CREATE with this visibility,
    #  but can_view_document() allows viewing)
    if assigned_client_ids:
        should.append(
            Filter(
                must=[
                    FieldCondition(
                        key="visibility", match=MatchValue(value=DocumentVisibility.CLIENT_PRIVATE)
                    ),
                    FieldCondition(key="owner_id", match=MatchAny(any=assigned_client_ids)),
                ]
            )
        )

    return Filter(should=should)
