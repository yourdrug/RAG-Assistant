"""
infrastructure/acl.py — Unified document access control rules.

Model:
  user kind:  internal | client
  document visibility:
    internal_public   — visible to all internal employees
    internal_group    — visible to group_id members (internal only)
    internal_private   — visible only to owner_id (internal)
    client_private     — visible to owner_id (client) + internal employees
                          assigned via client_assignments

client never sees internal_*, and vice versa — these visibility value sets
don't overlap, so mixing them up in a filter is structurally impossible.

Used in:
  - services/document_service.py        — validate_visibility() before writing documents
  - services/chat_service.py:rag_stream — build_qdrant_filter() before Qdrant search
"""

from config import settings
from fastapi import HTTPException
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, PayloadSchemaType
from sqlalchemy.orm import Session

from infrastructure.database import get_assigned_client_ids, get_user_group_ids

ALLOWED_VISIBILITY_FOR_KIND = {
    "internal": {"internal_public", "internal_group", "internal_private"},
    "client": {"client_private"},
}


def validate_visibility(visibility: str, group_id: int | None, current_user: dict, db: Session) -> None:
    allowed = ALLOWED_VISIBILITY_FOR_KIND.get(current_user["kind"])
    if allowed is None or visibility not in allowed:
        raise HTTPException(400, f"visibility='{visibility}' not available for kind='{current_user['kind']}'")

    if visibility == "internal_public" and current_user["role"] != "admin":
        raise HTTPException(403, "Only admin can publish to internal_public")

    if visibility == "internal_group":
        if group_id is None:
            raise HTTPException(400, "group_id required for visibility='internal_group'")
        if group_id not in get_user_group_ids(db, current_user["id"]):
            raise HTTPException(403, "You are not a member of this group")


def owner_and_group_for(
    visibility: str, group_id: int | None, current_user: dict
) -> tuple[int | None, int | None]:
    if visibility == "internal_public":
        return None, None
    if visibility == "internal_group":
        return None, group_id
    return current_user["id"], None


def can_view_document(db: Session, user: dict, doc: dict) -> bool:
    if doc["visibility"] == "internal_public":
        return user["kind"] == "internal"
    if doc["visibility"] == "internal_group":
        return user["kind"] == "internal" and doc["group_id"] in get_user_group_ids(db, user["id"])
    if doc["visibility"] == "internal_private":
        return user["kind"] == "internal" and doc["owner_id"] == user["id"]
    if doc["visibility"] == "client_private":
        if user["kind"] == "client":
            return doc["owner_id"] == user["id"]
        return doc["owner_id"] in get_assigned_client_ids(db, user["id"])
    return False


def build_qdrant_filter(user: dict, db: Session) -> Filter:
    if user["kind"] == "client":
        return Filter(
            must=[
                FieldCondition(key="visibility", match=MatchValue(value="client_private")),
                FieldCondition(key="owner_id", match=MatchValue(value=user["id"])),
            ]
        )

    group_ids = get_user_group_ids(db, user["id"])
    assigned_client_ids = get_assigned_client_ids(db, user["id"])

    should: list[FieldCondition | Filter] = [
        FieldCondition(key="visibility", match=MatchValue(value="internal_public")),
        Filter(
            must=[
                FieldCondition(key="visibility", match=MatchValue(value="internal_private")),
                FieldCondition(key="owner_id", match=MatchValue(value=user["id"])),
            ]
        ),
    ]
    if group_ids:
        should.append(
            Filter(
                must=[
                    FieldCondition(key="visibility", match=MatchValue(value="internal_group")),
                    FieldCondition(key="group_id", match=MatchAny(any=group_ids)),
                ]
            )
        )
    if assigned_client_ids:
        should.append(
            Filter(
                must=[
                    FieldCondition(key="visibility", match=MatchValue(value="client_private")),
                    FieldCondition(key="owner_id", match=MatchAny(any=assigned_client_ids)),
                ]
            )
        )
    return Filter(should=should)


def ensure_acl_payload_indexes(client) -> None:
    """Kept for backward compatibility. Prefer ClientContainer.ensure_acl_indexes()."""
    for field, schema in [
        ("visibility", PayloadSchemaType.KEYWORD),
        ("owner_id", PayloadSchemaType.INTEGER),
        ("group_id", PayloadSchemaType.INTEGER),
        ("document_id", PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(settings.collection_name, field_name=field, field_schema=schema)
        except Exception:
            pass
