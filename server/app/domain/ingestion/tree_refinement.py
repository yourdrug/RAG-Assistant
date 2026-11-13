"""Refine format nodes using validated profile boundaries, independently of budgets."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from domain.ingestion.content_splitter import order_enclosing_levels, split_at_boundary
from domain.ingestion.document_tree import DocNode, NodeKind, structural_kind, tree_text


if TYPE_CHECKING:
    from collections.abc import Callable

    from domain.ingestion.document_tree import BoundaryEventSink
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.profiles.types import BoundaryLevel


def refine_tree(
    tree: DocNode, profile: DocumentProfile, on_boundary_event: BoundaryEventSink | None = None
) -> DocNode:
    levels = profile.content_boundaries()

    if not levels or tree.kind == NodeKind.TABLE:
        return tree

    if tree.children:
        return replace(
            tree, children=tuple(refine_tree(child, profile, on_boundary_event) for child in tree.children)
        )

    metadata = tree.hints.get("metadata", {})
    tree = replace(
        tree,
        hints={
            **tree.hints,
            "line_hints": tree.hints.get("line_hints", metadata.get("line_hints", [])),
            "page_spans": tree.hints.get("page_spans", metadata.get("_page_spans", [])),
        },
    )
    return refine_text(tree, levels, profile.unit_label, on_event=on_boundary_event)


def refine_text(
    tree: DocNode,
    levels: list[BoundaryLevel],
    unit_label: Callable[[str, str], str] | None = None,
    *,
    parent_number: str | None = None,
    on_event: Callable[[str, int], None] | None = None,
) -> DocNode:
    source = tree_text(tree)
    levels = order_enclosing_levels(source, levels)

    if not levels:
        return tree

    level, *rest = levels
    confirmed = frozenset(
        hint["start"]
        for hint in tree.hints.get("line_hints", [])
        if hint.get("heading") or hint.get("numbering_level") is not None
    )
    units = split_at_boundary(
        source,
        level,
        unit_label=unit_label,
        following_levels=rest,
        confirmed_starts=confirmed,
        parent_number=parent_number,
        on_boundary_event=(lambda rule, pos: on_event(rule, tree.span[0] + pos)) if on_event else None,
    )
    children = []
    offset = tree.span[0]

    for unit in units:
        hints = {
            **{key: value for key, value in tree.hints.items() if key != "metadata"},
            "source_text": unit.content,
            "unit_kind": unit.unit_kind,
            "page_spans": [
                (max(low, unit.start) - unit.start, min(high, unit.end) - unit.start, page)
                for low, high, page in tree.hints.get("page_spans", [])
                if low < unit.end and high > unit.start
            ],
            "level": level,
            "following_levels": rest,
            "line_hints": [
                {**hint, "start": hint["start"] - unit.start, "end": hint["end"] - unit.start}
                for hint in tree.hints.get("line_hints", [])
                if unit.start <= hint["start"] < unit.end
            ],
        }
        node = DocNode(
            structural_kind(unit.unit_kind),
            unit.heading,
            unit.boundary_value,
            unit.content,
            span=(offset + unit.start, offset + unit.end),
            pages=tuple(sorted({page for _, _, page in hints["page_spans"]})) or tree.pages,
            hints=hints,
        )

        if rest:
            number = (
                unit.boundary_value
                if unit.unit_kind in ("point", "subpoint_num", "section", "clause", "subclause")
                else parent_number
            )
            node = refine_text(node, rest, unit_label, parent_number=number, on_event=on_event)

        children.append(node)

    return replace(tree, text="", children=tuple(children))
