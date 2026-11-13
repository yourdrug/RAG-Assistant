"""Pack adjacent structural units without crossing their enclosing scope."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from domain.ingestion.content_splitter import SplitUnit


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class PackedUnit(SplitUnit):
    boundary_values: tuple[str, ...] = ()
    point_numbers: tuple[str, ...] = ()


def pack_units(  # noqa: C901, PLR0913 - profile packing and exact source measurement
    units: list[SplitUnit],
    budget: int,
    measure: Callable[[str], int],
    *,
    rank: Mapping[str, int] | None = None,
    min_size: int = 0,
    source: str | None = None,
    scope_boundaries: frozenset[str] = frozenset(),
    exact_measure: bool = False,
) -> list[PackedUnit]:
    """Prefer the coarsest break in a full window; keep fitting units atomic.

    A supplied source preserves exact whitespace and offsets, including repeated
    text. Without one, callers get newline-separated bodies. Oversized atoms are
    returned intact for the caller's finer splitter to handle. ``budget`` and
    ``min_size`` use the units returned by ``measure``. Profiles supply the
    levels that must remain separate through ``scope_boundaries``.
    """
    if budget <= 0 or min_size < 0:
        message = "budget must be positive and min_size non-negative"
        raise ValueError(message)

    sizes = {id(unit): measure(unit.content) for unit in units}
    gaps = {
        (id(left), id(right)): (
            measure(source[left.end : right.start])
            if source is not None and left.end < right.start
            else (measure("\n") if source is None else 0)
        )
        for left, right in pairwise(units)
    }

    def window_size(group: list[SplitUnit]) -> int:
        if exact_measure:
            return measure(packing_content(group, source))

        return sum(sizes[id(unit)] for unit in group) + sum(
            gaps.get((id(left), id(right)), 0) for left, right in pairwise(group)
        )

    ranks = rank or {}
    result: list[PackedUnit] = []
    pending: list[SplitUnit] = []

    for unit in units:
        barrier = pending and not same_packing_scope(pending[-1], unit, scope_boundaries)

        if barrier:
            result.append(packed_group(pending, source, ranks))
            pending = []

        while pending and window_size([*pending, unit]) > budget:
            window = [*pending, unit]
            cut = min(
                range(1, len(window)),
                key=lambda index: (ranks.get(window[index].unit_kind, len(ranks)), -index),
            )
            result.append(packed_group(window[:cut], source, ranks))
            pending = window[cut:-1]

        pending.append(unit)

    if pending:
        if measure(packing_content(pending, source)) < min_size:
            append_packing_tail(
                result, pending, source, ranks, measure, budget=budget, scope_boundaries=scope_boundaries
            )
        else:
            result.append(packed_group(pending, source, ranks))

    return result


def packing_content(group: list[SplitUnit], source: str | None) -> str:
    if source is not None:
        return source[group[0].start : group[-1].end]

    return "\n".join(unit.content for unit in group)


def packed_group(group: list[SplitUnit], source: str | None, ranks: Mapping[str, int]) -> PackedUnit:
    highest = min(group, key=lambda unit: ranks.get(unit.unit_kind, len(ranks)))
    values = tuple(value for unit in group for value in packing_boundary_values(unit))
    points = tuple(
        value for unit in group if unit.unit_kind == "point" for value in packing_boundary_values(unit)
    )
    return PackedUnit(
        heading=highest.heading,
        content=packing_content(group, source),
        unit_kind=highest.unit_kind,
        boundary_value=values[0] if len(values) == 1 else None,
        parents=highest.parents,
        start=group[0].start,
        end=group[-1].end,
        boundary_values=values,
        point_numbers=points,
    )


def append_packing_tail(
    result: list[PackedUnit],
    pending: list[SplitUnit],
    source: str | None,
    ranks: Mapping[str, int],
    measure: Callable[[str], int],
    *,
    budget: int,
    scope_boundaries: frozenset[str],
) -> None:
    """Join a small tail only when scope and the hard budget permit it."""
    tail = packed_group(pending, source, ranks)

    if result and same_packing_scope(result[-1], tail, scope_boundaries):
        combined = packed_group([result[-1], *pending], source, ranks)

        if measure(combined.content) <= budget:
            result[-1] = combined
            return

    result.append(tail)


def same_packing_scope(previous: SplitUnit, current: SplitUnit, scope_boundaries: frozenset[str]) -> bool:
    """Respect parent identity and the profile's enclosing boundaries."""
    return (
        previous.parents == current.parents
        and previous.unit_kind not in scope_boundaries
        and current.unit_kind not in scope_boundaries
    )


def packing_boundary_values(unit: SplitUnit) -> tuple[str, ...]:
    """Keep all numbers when a packed tail is joined a second time."""
    if isinstance(unit, PackedUnit):
        return unit.boundary_values

    return (unit.boundary_value,) if unit.boundary_value is not None else ()
