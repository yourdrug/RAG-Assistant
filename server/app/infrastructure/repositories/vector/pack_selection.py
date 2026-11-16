"""Translate a pack metadata selection to a Qdrant filter over ``metadata.pack_metadata``.

Mirrors ``domain.knowledge.pack_metadata.matches`` and the SQL translation, so vector and
substring retrieval select the same chunks before top-k.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchValue,
    PayloadField,
)

from domain.knowledge.pack_metadata import (
    PACK_METADATA_KEY,
    AllOf,
    AnyOf,
    DateAfter,
    DateOnOrBefore,
    Equals,
    Missing,
    Not,
)


if TYPE_CHECKING:
    from qdrant_client.models import Condition

    from domain.knowledge.pack_metadata import MetadataCondition


def payload_key(key: str) -> str:
    return f"metadata.{PACK_METADATA_KEY}.{key}"


def pack_selection_condition(condition: MetadataCondition) -> Condition:
    result: Condition
    match condition:
        case Missing(key):
            # IsEmptyCondition matches both absent and explicit-null payload keys.
            result = IsEmptyCondition(is_empty=PayloadField(key=payload_key(key)))
        case Equals(key, value):
            result = FieldCondition(key=payload_key(key), match=MatchValue(value=value))
        case DateOnOrBefore(key, value):
            result = FieldCondition(key=payload_key(key), range=DatetimeRange(lte=value))
        case DateAfter(key, value):
            result = FieldCondition(key=payload_key(key), range=DatetimeRange(gt=value))
        case AllOf(conditions):
            result = Filter(must=[pack_selection_condition(item) for item in conditions])
        case AnyOf(conditions):
            result = Filter(should=[pack_selection_condition(item) for item in conditions])
        case Not(inner):
            result = Filter(must_not=[pack_selection_condition(inner)])
        case _:
            assert_never(condition)

    return result
