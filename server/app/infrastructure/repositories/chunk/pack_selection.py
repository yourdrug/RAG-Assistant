"""Translate a pack metadata selection to a SQL predicate over ``chunks.pack_metadata``.

Mirrors ``domain.knowledge.pack_metadata.matches`` and the Qdrant translation, so substring
retrieval and vector retrieval select the same chunks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from sqlalchemy import Date, and_, cast, false, func, not_, or_, true

from domain.knowledge.pack_metadata import AllOf, AnyOf, DateAfter, DateOnOrBefore, Equals, Missing, Not
from infrastructure.database.models import ChunkModel


if TYPE_CHECKING:
    from datetime import date

    from sqlalchemy.sql.elements import ColumnElement

    from domain.knowledge.pack_metadata import MetadataCondition


JSON_TYPES = {bool: "boolean", int: "number", str: "string"}


def present(key: str) -> ColumnElement[bool]:
    clause: ColumnElement[bool] = ChunkModel.pack_metadata[key].astext.is_not(None)
    return clause


def iso_date(key: str) -> ColumnElement[date]:
    return cast(func.left(ChunkModel.pack_metadata[key].astext, 10), Date())


def leaf_clause(condition: Missing | Equals | DateOnOrBefore | DateAfter) -> ColumnElement[bool]:
    clause: ColumnElement[bool]
    match condition:
        case Missing(key):
            clause = ChunkModel.pack_metadata[key].astext.is_(None)
        case Equals(key, value):
            text_value = str(value).lower() if isinstance(value, bool) else str(value)
            clause = and_(
                func.jsonb_typeof(ChunkModel.pack_metadata[key]) == JSON_TYPES[type(value)],
                ChunkModel.pack_metadata[key].astext == text_value,
            )
        case DateOnOrBefore(key, value):
            clause = and_(present(key), iso_date(key) <= value)
        case DateAfter(key, value):
            clause = and_(present(key), iso_date(key) > value)
        case _:
            assert_never(condition)

    return clause


def pack_selection_clause(condition: MetadataCondition | None) -> ColumnElement[bool]:
    clause: ColumnElement[bool]
    match condition:
        case None:
            clause = true()
        case AllOf(conditions):
            clause = and_(true(), *(pack_selection_clause(item) for item in conditions))
        case AnyOf(conditions):
            clause = or_(false(), *(pack_selection_clause(item) for item in conditions))
        case Not(inner):
            clause = not_(pack_selection_clause(inner))
        case _:
            clause = leaf_clause(condition)

    return clause
