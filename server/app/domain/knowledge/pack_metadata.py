"""Chunk metadata owned by a domain pack and the neutral selection conditions over it.

The core stores and carries ``pack_metadata`` (SQL, vector payload, retrieved documents)
without interpreting it. A pack interprets its own keys and may narrow retrieval with a
``MetadataCondition``; adapters translate one condition to every store, so the selection
is the same in Qdrant and SQL. A selection chooses documents; it never grants access.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, assert_never


if TYPE_CHECKING:
    from collections.abc import Mapping


# Metadata key holding the pack-owned dictionary in chunk payloads and documents.
PACK_METADATA_KEY = "pack_metadata"


@dataclass(frozen=True)
class Missing:
    """The key is absent or null."""

    key: str


@dataclass(frozen=True)
class Equals:
    """Typed equality; a pack keeps one JSON type per key (stores compare booleans and integers loosely)."""

    key: str
    value: bool | int | str


@dataclass(frozen=True)
class DateOnOrBefore:
    """An ISO date value not later than ``value``; a missing key does not match."""

    key: str
    value: date


@dataclass(frozen=True)
class DateAfter:
    """An ISO date value strictly later than ``value``; a missing key does not match."""

    key: str
    value: date


@dataclass(frozen=True)
class AllOf:
    conditions: tuple[MetadataCondition, ...]


@dataclass(frozen=True)
class AnyOf:
    conditions: tuple[MetadataCondition, ...]


@dataclass(frozen=True)
class Not:
    condition: MetadataCondition


type MetadataCondition = Missing | Equals | DateOnOrBefore | DateAfter | AllOf | AnyOf | Not


def pack_metadata_of(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Pack-owned metadata of a retrieved chunk; empty when the chunk has none."""
    value = metadata.get(PACK_METADATA_KEY)
    return value if isinstance(value, dict) else {}


def parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None

    return None


def matches(condition: MetadataCondition, pack_metadata: Mapping[str, Any]) -> bool:
    """Reference semantics the store adapters must reproduce."""
    result: bool
    match condition:
        case Missing(key):
            result = pack_metadata.get(key) is None
        case Equals(key, value):
            stored = pack_metadata.get(key)
            result = stored is not None and type(stored) is type(value) and stored == value
        case DateOnOrBefore(key, value):
            stored_date = parse_date(pack_metadata.get(key))
            result = stored_date is not None and stored_date <= value
        case DateAfter(key, value):
            stored_date = parse_date(pack_metadata.get(key))
            result = stored_date is not None and stored_date > value
        case AllOf(conditions):
            result = all(matches(item, pack_metadata) for item in conditions)
        case AnyOf(conditions):
            result = any(matches(item, pack_metadata) for item in conditions)
        case Not(inner):
            result = not matches(inner, pack_metadata)
        case _:
            assert_never(condition)

    return result
