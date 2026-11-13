"""One depth-first packer for document trees and profile-owned policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import groupby
from typing import TYPE_CHECKING, Protocol

from domain.ingestion.chunking_policy import ChunkingPolicy
from domain.ingestion.content_splitter import (
    BoundaryContext,
    SplitUnit,
    heading_only_intro,
    merge_small_units,
    needs_finer_boundary,
    unit_parent_context,
)
from domain.ingestion.document_tree import DocNode, NodeKind, structural_kind, tree_text, walk_tree
from domain.ingestion.embed_text import (
    EMBED_PREFIX_KEY,
    MAX_EMBED_TOKENS,
    build_embed_text,
    context_prefix,
)
from domain.ingestion.text_budget import locate_span, overlap_tail, split_leaf_text
from domain.ingestion.tree_refinement import refine_text
from domain.ingestion.unit_packing import pack_units, packed_group


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from domain.ingestion.profiles.types import BoundaryLevel
    from domain.ingestion.unit_packing import PackedUnit


class UnitBudget(Protocol):
    def __call__(self, unit: SplitUnit, *, target: bool) -> int: ...


@dataclass
class TreeSelector:
    tree: DocNode
    policy: ChunkingPolicy
    measure: Callable[[str], int]
    budget: UnitBudget | None
    selected: list[SplitUnit] = field(default_factory=list)
    parents_by_start: dict[int, str | None] = field(default_factory=dict)
    pages_by_start: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return tree_text(self.tree)

    def visit(self, node: DocNode, parents: tuple[BoundaryContext, ...], parent_id: str | None) -> None:
        content = self.source[node.span[0] - self.tree.span[0] : node.span[1] - self.tree.span[0]]
        unit = SplitUnit(
            node.label,
            content,
            node.hints.get("unit_kind", node.kind.value),
            node.number,
            parents,
            *node.span,
        )
        level = node.hints.get("level")
        rest = node.hints.get("following_levels", [])
        limit = (
            min(self.policy.max_tokens, self.budget(unit, target=False))
            if self.budget
            else self.policy.max_tokens
        )
        descend = node.kind == NodeKind.DOCUMENT or (level is None and self.measure(content) > limit)

        if level is not None and rest:
            descend = needs_finer_boundary(unit, level, rest, limit, self.measure)

        descend = descend or any(child.kind == NodeKind.TABLE for child in walk_tree(node)[1:])

        if node.children and descend:
            child_parents = node_parent_context(node, unit, parents)
            first_index = len(self.selected)

            for child in node.children:
                self.visit(child, child_parents, node.node_id if node.label else parent_id)

            if len(self.selected) > first_index and node.label and level is not None:
                first = self.selected[first_index]

                if discard_heading_intro(first, node.label, level):
                    self.selected.pop(first_index)
        elif content.strip():
            self.selected.append(unit)
            self.parents_by_start[unit.start] = parent_id
            self.pages_by_start[unit.start] = node.pages


def node_parent_context(
    node: DocNode,
    unit: SplitUnit,
    parents: tuple[BoundaryContext, ...],
) -> tuple[BoundaryContext, ...]:
    level = node.hints.get("level")

    if node.label and level is not None and node.kind != NodeKind.SENTENCE:
        return (*parents, unit_parent_context(unit, node.hints.get("following_levels", [])))

    if node.label and level is None:
        return (*parents, BoundaryContext(node.label, node.kind.value, node.number, node.text))

    return parents


@dataclass(frozen=True)
class ChunkDraft:
    content: str
    embed_text: str
    path: tuple[str, ...]
    span: tuple[int, int]
    pages: tuple[int, ...]
    unit_kind: NodeKind
    numbers: tuple[str, ...]
    parent_node_id: str | None
    unit: PackedUnit
    metadata: Mapping[str, Any] = field(default_factory=dict)


def discard_heading_intro(unit: SplitUnit, label: str, level: BoundaryLevel) -> bool:
    return (unit.unit_kind == "preamble" and heading_only_intro(unit.content, level)) or (
        unit.unit_kind in ("preamble", "sentence")
        and level.packing_barrier
        and bool(level.pattern.match(unit.content))
        and " ".join(label.split()).startswith(" ".join(unit.content.split()))
    )


def pack_tree(
    tree: DocNode,
    policy: ChunkingPolicy,
    measure: Callable[[str], int],
    *,
    budget: UnitBudget | None = None,
    split_leaves: bool = True,
    on_fallback: Callable[[], None] | None = None,
    table_packer: Callable[[DocNode], list[ChunkDraft]] | None = None,
) -> list[ChunkDraft]:
    """Keep fitting atoms, descend oversized nodes and pack only within scope.

    The optional budget reserves the same context used by the embedding adapter.
    Table leaves remain atomic for the row/header adapter.
    """
    if tree.kind == NodeKind.TABLE and table_packer is not None:
        return table_packer(tree)

    source = tree_text(tree)
    selector = TreeSelector(tree, policy, measure, budget)
    selector.visit(tree, (), None)
    selected = selector.selected
    levels = [node.hints["level"] for node in walk_levels(tree) if "level" in node.hints]
    rank = {level.name: i for i, level in enumerate({level.name: level for level in levels}.values())}
    barriers = frozenset({NodeKind.TABLE.value, *(level.name for level in levels if level.packing_barrier)})
    packed = []
    # pack_units source offsets are document-global; retain exact source gaps.
    padded_source = " " * tree.span[0] + source

    for _, siblings in groupby(
        selected, key=lambda unit: (unit.parents, selector.parents_by_start.get(unit.start))
    ):
        group = list(siblings)
        limit = min(budget(unit, target=True) for unit in group) if budget else policy.target_tokens
        packed.extend(
            pack_units(
                group,
                limit,
                measure,
                rank=rank,
                min_size=policy.min_tokens,
                source=padded_source,
                scope_boundaries=barriers,
                exact_measure=not levels,
            )
            if policy.pack_siblings
            else [packed_group([unit], padded_source, rank) for unit in group]
        )

    drafts = [
        ChunkDraft(
            unit.content,
            unit.content,
            tuple(parent.heading for parent in unit.parents) + ((unit.heading,) if unit.heading else ()),
            (unit.start, unit.end),
            tuple(
                sorted(
                    {
                        page
                        for item in selected
                        if unit.start <= item.start < unit.end
                        for page in selector.pages_by_start.get(item.start, ())
                    }
                )
            ),
            structural_kind(unit.unit_kind),
            unit.boundary_values,
            selector.parents_by_start.get(unit.start),
            unit,
        )
        for unit in packed
    ]

    if table_packer is not None:
        drafts = route_tables(drafts, tree, table_packer)

    if split_leaves:
        drafts = [piece for draft in drafts for piece in fit_draft(draft, policy, measure, on_fallback)]

    return [with_embedding(draft, tree, policy, measure) for draft in drafts] if split_leaves else drafts


def route_tables(
    drafts: list[ChunkDraft],
    tree: DocNode,
    table_packer: Callable[[DocNode], list[ChunkDraft]],
) -> list[ChunkDraft]:
    tables = {node.span: node for node in walk_tree(tree) if node.kind == NodeKind.TABLE}
    result = []

    for draft in drafts:
        if draft.unit_kind != NodeKind.TABLE:
            result.append(draft)
            continue

        node = tables[draft.span]
        result.extend(
            replace(piece, path=draft.path, parent_node_id=draft.parent_node_id)
            for piece in table_packer(node)
        )

    return result


def with_embedding(
    draft: ChunkDraft,
    tree: DocNode,
    policy: ChunkingPolicy,
    measure: Callable[[str], int],
) -> ChunkDraft:
    metadata = dict(tree.hints.get("metadata", {}))

    if not policy.include_doc_title:
        metadata.pop("doc_title", None)

    metadata["section"] = " > ".join(draft.path[-policy.context_depth :]) if policy.context_depth else ""
    metadata["parent_units"] = (
        [asdict(parent) for parent in draft.unit.parents[-policy.context_depth :]]
        if policy.context_depth
        else []
    )
    metadata[EMBED_PREFIX_KEY] = context_prefix(
        metadata,
        policy.max_tokens,
        body=draft.content,
        measure=measure,
    )
    dense = build_embed_text(metadata, draft.content, measure=measure, max_tokens=MAX_EMBED_TOKENS)
    return replace(draft, embed_text=dense)


def fit_draft(
    draft: ChunkDraft,
    policy: ChunkingPolicy,
    measure: Callable[[str], int],
    on_fallback: Callable[[], None] | None,
) -> list[ChunkDraft]:
    if measure(draft.content) <= policy.max_tokens or draft.unit_kind == NodeKind.TABLE:
        return [draft]

    bodies = split_leaf_text(
        draft.content,
        policy.target_tokens,
        policy.overlap_tokens,
        measure=measure,
        on_fallback=on_fallback,
    )
    result = []
    cursor = 0

    for body in bodies:
        located = locate_span(draft.content, body, cursor)

        if located is None:
            message = "Cannot locate chunk in source node"
            raise ValueError(message)

        start, end = located
        cursor = max(start + 1, end - len(overlap_tail(body, policy.overlap_tokens, measure)))
        unit = replace(draft.unit, content=body, start=draft.span[0] + start, end=draft.span[0] + end)
        result.append(replace(draft, content=body, embed_text=body, span=(unit.start, unit.end), unit=unit))

    return result


def walk_levels(tree: DocNode) -> list[DocNode]:
    return [tree, *(node for child in tree.children for node in walk_levels(child))]


@dataclass(frozen=True)
class ProjectionBudget:
    measure_unit: Callable[[SplitUnit], int]

    def __call__(self, unit: SplitUnit, *, target: bool) -> int:
        del target
        return self.measure_unit(unit)


def select_structural_units(  # noqa: PLR0913 - compatibility projection of the tree for RTF and fixtures
    text: str,
    levels: list[BoundaryLevel],
    max_unit_size: int,
    min_chunk_chars: int = 0,
    *,
    unit_budget: Callable[[SplitUnit], int] | None = None,
    unit_label: Callable[[str, str], str] | None = None,
    measure: Callable[[str], int] = len,
    on_boundary_event: Callable[[str, int], None] | None = None,
    confirmed_starts: frozenset[int] = frozenset(),
) -> list[SplitUnit]:
    tree = DocNode(
        NodeKind.DOCUMENT if levels else NodeKind.RAW,
        text=text,
        span=(0, len(text)),
        hints={
            "source_text": text,
            "line_hints": [{"start": start, "end": start, "heading": True} for start in confirmed_starts],
        },
    )
    tree = refine_text(tree, levels, unit_label, on_event=on_boundary_event)
    selector = TreeSelector(
        tree,
        ChunkingPolicy(max_unit_size, max_unit_size),
        measure,
        ProjectionBudget(unit_budget) if unit_budget else None,
    )
    selector.visit(tree, (), None)
    units = merge_small_units(selector.selected, min_chunk_chars)
    return [replace(unit, content=text[unit.start : unit.end]) for unit in units]
