"""Immutable group query results."""

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
