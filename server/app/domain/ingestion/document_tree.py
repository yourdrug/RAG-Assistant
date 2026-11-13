"""Format-independent document structure with offsets in normalized source text."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


# Receives (rule, absolute offset) for rejected/recovered boundary candidates.
type BoundaryEventSink = Callable[[str, int], None]


class NodeKind(StrEnum):
    STRUCTURAL = "structural"
    DOCUMENT = "document"
    HEADING_BLOCK = "heading_block"
    CHAPTER = "chapter"
    SECTION = "section"
    APPENDIX = "appendix"
    ARTICLE = "article"
    POINT = "point"
    SUBPOINT_NUM = "subpoint_num"
    SUBPOINT = "subpoint"
    CLAUSE = "clause"
    SUBCLAUSE = "subclause"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    PREAMBLE = "preamble"
    RAW = "raw"
    TABLE = "table"
    TABLE_ROW = "table_row"
    LIST = "list"
    STEP = "step"


@dataclass(frozen=True)
class DocNode:
    kind: NodeKind
    label: str | None = None
    number: str | None = None
    text: str = ""
    children: tuple[DocNode, ...] = ()
    span: tuple[int, int] = (0, 0)
    pages: tuple[int, ...] = ()
    hints: Mapping[str, Any] = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        """Document-local identity; storage must qualify it with document/version."""
        return f"{self.kind.value}:{self.span[0]}:{self.span[1]}"


def walk_tree(tree: DocNode) -> tuple[DocNode, ...]:
    return (tree, *(node for child in tree.children for node in walk_tree(child)))


def tree_text(tree: DocNode) -> str:
    """Recover exact normalized text, including whitespace between children."""
    source = tree.hints.get("source_text")

    if isinstance(source, str):
        return source

    return tree.text + "".join(tree_text(child) for child in tree.children)


def structural_kind(value: str) -> NodeKind:
    """Custom profiles retain their vocabulary without adding string node kinds."""
    return next((kind for kind in NodeKind if kind.value == value), NodeKind.STRUCTURAL)
