"""Owner match type for ACL visibility conditions."""

from __future__ import annotations

from enum import StrEnum


class OwnerMatch(StrEnum):
    SELF = "self"
    ASSIGNED = "assigned"
