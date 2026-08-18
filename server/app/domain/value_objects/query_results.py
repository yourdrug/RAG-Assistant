"""Domain value objects for repository query results.

Replaces untyped ``dict`` returns with typed, immutable data objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupInfo:
    """Lightweight group representation returned by group queries."""

    id: int
    name: str


@dataclass(frozen=True)
class GroupMemberInfo:
    """Group member returned by list_members."""

    id: int
    email: str


@dataclass(frozen=True)
class ApiKeyClientInfo:
    """Combined ApiKey + User data returned by get_active_client_by_hash."""

    api_key_id: int
    id: int
    email: str
    role: str
    kind: str
    is_active: bool
