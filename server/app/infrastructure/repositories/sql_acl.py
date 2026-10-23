"""Translate domain visibility conditions into SQL AND-clauses for callers to OR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import and_

from domain.knowledge.access_control import OwnerMatch


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import InstrumentedAttribute
    from sqlalchemy.sql.elements import ColumnElement

    from domain.knowledge.access_control import VisibilityCondition


def render_visibility_conditions(
    conditions: Sequence[VisibilityCondition],
    *,
    visibility_column: InstrumentedAttribute[str],
    owner_column: InstrumentedAttribute[int | None],
    group_column: InstrumentedAttribute[int | None],
    user_id: int,
    group_ids: list[int],
) -> list[ColumnElement[bool]]:
    """Render prepared conditions without selecting listing or search policy."""
    clauses = []

    for condition in conditions:
        parts = [visibility_column == condition.visibility.value]

        if condition.owner_match == OwnerMatch.SELF:
            parts.append(owner_column == user_id)

        if condition.owner_match == OwnerMatch.ASSIGNED and condition.owner_ids:
            parts.append(owner_column.in_(condition.owner_ids))

        if condition.group_match:
            effective_group_ids = condition.group_ids if condition.group_ids is not None else group_ids
            parts.append(group_column.in_(effective_group_ids))

        clauses.append(and_(*parts))

    return clauses
