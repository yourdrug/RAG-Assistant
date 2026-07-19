"""
infrastructure/acl.py — единая точка правил доступа к документам.

Модель:
  kind пользователя:  internal | client
  visibility документа:
    internal_public   — видят все internal-сотрудники
    internal_group    — видят участники group_id (только internal)
    internal_private   — видит только owner_id (internal)
    client_private     — видит owner_id (client) + internal-сотрудники,
                          которым этот клиент назначен через client_assignments

client никогда не видит internal_*, и наоборот — эти множества значений visibility
не пересекаются, так что перепутать их в фильтре структурно невозможно.

Используется в двух местах:
  - api/routes.py                        — validate_visibility() перед записью документа
  - infrastructure/vector_store.py:rag_stream — build_qdrant_filter() перед поиском в Qdrant
"""

from fastapi import HTTPException
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from sqlalchemy.orm import Session

from infrastructure.database import get_assigned_client_ids, get_user_group_ids

ALLOWED_VISIBILITY_FOR_KIND = {
    "internal": {"internal_public", "internal_group", "internal_private"},
    "client": {"client_private"},
}


def validate_visibility(visibility: str, group_id: int | None, current_user: dict, db: Session) -> None:
    """Бросает HTTPException, если пользователю нельзя публиковать документ с такой
    видимостью. owner_id/group_id, которые реально попадут в БД и Qdrant, вычисляются
    отдельно через owner_and_group_for() — эта функция только проверяет допустимость.
    """
    allowed = ALLOWED_VISIBILITY_FOR_KIND.get(current_user["kind"])
    if allowed is None or visibility not in allowed:
        raise HTTPException(400, f"visibility='{visibility}' недоступна для kind='{current_user['kind']}'")

    if visibility == "internal_public" and current_user["role"] != "admin":
        raise HTTPException(403, "Публиковать в internal_public может только admin")

    if visibility == "internal_group":
        if group_id is None:
            raise HTTPException(400, "Для visibility='internal_group' нужен group_id")
        if group_id not in get_user_group_ids(db, current_user["id"]):
            raise HTTPException(403, "Вы не состоите в этой группе")


def owner_and_group_for(
    visibility: str, group_id: int | None, current_user: dict
) -> tuple[int | None, int | None]:
    """Вычисляет owner_id/group_id для БД и payload Qdrant.
    Вызывать ТОЛЬКО после успешного validate_visibility()."""
    if visibility == "internal_public":
        return None, None
    if visibility == "internal_group":
        return None, group_id
    return current_user["id"], None


def can_view_document(db: Session, user: dict, doc: dict) -> bool:
    """Проверка доступа к одному документу (GET /documents/{id})."""
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
    """Строит Qdrant Filter, ограничивающий поиск документами, которые видны
    данному пользователю. Применяется ДО реранка.
    """
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


_ACL_INDEXES_ENSURED = False


def ensure_acl_payload_indexes(client) -> None:
    global _ACL_INDEXES_ENSURED
    if _ACL_INDEXES_ENSURED:
        return
    from config import settings
    from qdrant_client.models import PayloadSchemaType

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
    _ACL_INDEXES_ENSURED = True
