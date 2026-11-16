"""Shared pack metadata and selections: every store must agree with the reference semantics."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from domain.knowledge.pack_metadata import AllOf, AnyOf, DateAfter, DateOnOrBefore, Equals, Missing, Not


if TYPE_CHECKING:
    from domain.knowledge.pack_metadata import MetadataCondition


DAY = date(2025, 6, 1)

# Index = point/chunk id offset; covers absent, null, typed and dated values.
# A pack keeps one JSON type per key, so a key never mixes booleans and integers.
PACK_METADATA: list[dict[str, Any]] = [
    {},
    {"flag": None, "valid_from": None},
    {"flag": True},
    {"flag": False},
    {"count": 1},
    {"flag": "true"},
    {"kind": "faq"},
    {"kind": "guide", "valid_from": "2025-06-01"},
    {"valid_from": "2025-05-31", "valid_to": "2025-06-01"},
    {"valid_from": "2025-06-02", "valid_to": "2026-01-01"},
    {"valid_to": "2025-06-02", "flag": True},
]

SELECTIONS: dict[str, MetadataCondition] = {
    "missing": Missing("flag"),
    "equals_true": Equals("flag", value=True),
    "equals_int": Equals("count", 1),
    "equals_str": Equals("kind", "faq"),
    "date_on_or_before": DateOnOrBefore("valid_from", DAY),
    "date_after": DateAfter("valid_to", DAY),
    "not_missing": Not(Missing("valid_from")),
    "interval": AllOf(
        (
            AnyOf((Missing("valid_from"), DateOnOrBefore("valid_from", DAY))),
            AnyOf((Missing("valid_to"), DateAfter("valid_to", DAY))),
        )
    ),
    "empty_any": AnyOf(()),
    "empty_all": AllOf(()),
}
