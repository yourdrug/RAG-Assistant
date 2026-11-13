"""Build the common domain tree from format parsers; bridge it to legacy sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from domain.ingestion.document_tree import DocNode, NodeKind, tree_text, walk_tree
from domain.knowledge.enums import PageContentType
from infrastructure.ml.ingestion.section_types import Section


if TYPE_CHECKING:
    from collections.abc import Iterable


BLOCK_SEPARATOR = "\n\n"


@dataclass
class PendingHeading:
    level: int
    label: str
    emit: bool
    children: list[PendingHeading | PendingBlock] = field(default_factory=list)


@dataclass(frozen=True)
class PendingBlock:
    text: str
    kind: NodeKind
    metadata: dict[str, Any]
    pages: tuple[int, ...]


class DocumentTreeBuilder:
    """Collect parser headings and blocks in document order, then lay out one source text.

    Hierarchy comes from heading levels reported by the format (styles, numbering,
    TOC, fonts), never from re-parsing breadcrumb strings. A heading whose label is
    not part of any block (``emit=True``) contributes its own line to the source
    text, so the root ``source_text`` is the whole normalized document and every
    node ``span`` indexes it.
    """

    def __init__(self) -> None:
        self.root = PendingHeading(-1, "", emit=False)
        self.stack = [self.root]

    def heading(self, level: int, label: str, *, emit: bool = True) -> None:
        """Open a heading, closing open headings of the same or a deeper level."""
        while len(self.stack) > 1 and self.stack[-1].level >= level:
            self.stack.pop()

        node = PendingHeading(level, label, emit)
        self.stack[-1].children.append(node)
        self.stack.append(node)

    def close_headings(self) -> None:
        self.stack = [self.root]

    def block(
        self,
        text: str,
        kind: NodeKind = NodeKind.PARAGRAPH,
        metadata: dict[str, Any] | None = None,
        pages: Iterable[int] = (),
    ) -> None:
        self.stack[-1].children.append(PendingBlock(text, kind, dict(metadata or {}), tuple(pages)))

    def has_blocks(self) -> bool:
        return any(isinstance(node, PendingBlock) for node in pending_walk(self.root))

    def build(self) -> DocNode:
        parts: list[str] = []
        page_spans: list[tuple[int, int, int]] = []
        offset = 0

        def write(text: str) -> tuple[int, int]:
            nonlocal offset

            if parts:
                parts.append(BLOCK_SEPARATOR)
                offset += len(BLOCK_SEPARATOR)

            parts.append(text)
            offset += len(text)
            return offset - len(text), offset

        def freeze(node: PendingHeading | PendingBlock, path: tuple[str, ...]) -> DocNode:
            if isinstance(node, PendingBlock):
                span = write(node.text)
                pages = node.pages or block_pages(node.metadata)

                if len(pages) == 1:
                    page_spans.append((*span, pages[0]))

                return DocNode(
                    node.kind,
                    text=node.text,
                    span=span,
                    pages=pages,
                    hints={"metadata": node.metadata, "heading_path": path},
                )

            if node is not self.root:
                path = (*path, node.label)

            own = write(node.label) if node.emit else (offset, offset)
            children = tuple(freeze(child, path) for child in node.children)
            span = (own[0], children[-1].span[1] if children else own[1])
            return DocNode(
                NodeKind.HEADING_BLOCK if node is not self.root else NodeKind.DOCUMENT,
                label=node.label or None,
                text=node.label if node.emit else "",
                children=children,
                span=span,
                pages=tuple(sorted({page for child in children for page in child.pages})),
                hints={"heading_path": path, "level": node.level},
            )

        tree = freeze(self.root, ())
        source = "".join(parts)
        return attach_source_text(tree, source, page_spans)


def pending_walk(node: PendingHeading | PendingBlock) -> list[PendingHeading | PendingBlock]:
    if isinstance(node, PendingBlock):
        return [node]

    return [node, *(item for child in node.children for item in pending_walk(child))]


def block_pages(metadata: dict[str, Any]) -> tuple[int, ...]:
    return tuple(metadata.get("pages") or ([metadata["page"]] if "page" in metadata else []))


def attach_source_text(tree: DocNode, source: str, page_spans: list[tuple[int, int, int]]) -> DocNode:
    """Branches carry their exact source slice so ``tree_text`` keeps separators."""
    children = tuple(
        child if "metadata" in child.hints else attach_source_text(child, source, [])
        for child in tree.children
    )
    hints = {**tree.hints, "source_text": source[tree.span[0] : tree.span[1]]}

    if tree.kind == NodeKind.DOCUMENT:
        hints = {**hints, "source_text": source, "page_spans": page_spans}
        return DocNode(
            tree.kind, tree.label, tree.number, tree.text, children, (0, len(source)), tree.pages, hints
        )

    return DocNode(tree.kind, tree.label, tree.number, tree.text, children, tree.span, tree.pages, hints)


def single_block_tree(text: str) -> DocNode:
    builder = DocumentTreeBuilder()
    builder.block(text)
    return builder.build()


def content_blocks(tree: DocNode) -> list[DocNode]:
    return [node for node in walk_tree(tree) if "metadata" in node.hints]


def has_heading_scopes(tree: DocNode) -> bool:
    return any(node.hints["heading_path"] for node in content_blocks(tree))


def tree_from_sections(sections: list[Section]) -> DocNode:
    source = "\n\n".join(section.content for section in sections)
    # Mutable builders are local; the published tree consists of frozen nodes.
    root: dict[str, Any] = {"label": None, "children": [], "path": ()}
    stack = [root]
    offset = 0

    for section in sections:
        path = tuple(section.heading.split(" > ")) if section.heading else ()
        common = 0

        while common < min(len(path), len(stack) - 1) and stack[common + 1]["label"] == path[common]:
            common += 1

        stack = stack[: common + 1]

        for label in path[common:]:
            branch = {"label": label, "children": [], "path": (*stack[-1]["path"], label)}
            stack[-1]["children"].append(branch)
            stack.append(branch)

        meta = dict(section.metadata)
        pages = tuple(meta.get("pages") or ([meta["page"]] if "page" in meta else []))
        stack[-1]["children"].append(
            DocNode(
                NodeKind.TABLE if section.kind == PageContentType.TABLE else NodeKind.PARAGRAPH,
                text=section.content,
                span=(offset, offset + len(section.content)),
                pages=pages,
                hints={"metadata": meta, "heading_path": path, "source_text": section.content},
            )
        )
        offset += len(section.content) + 2

    def freeze(branch: dict[str, Any]) -> DocNode:
        children = tuple(
            child if isinstance(child, DocNode) else freeze(child) for child in branch["children"]
        )
        span = (children[0].span[0], children[-1].span[1]) if children else (0, 0)
        return DocNode(
            NodeKind.HEADING_BLOCK if branch["label"] else NodeKind.DOCUMENT,
            label=branch["label"],
            children=children,
            span=span,
            pages=tuple(sorted({page for child in children for page in child.pages})),
            hints={"heading_path": branch["path"], "source_text": source[span[0] : span[1]]},
        )

    return freeze(root)


def sections_from_tree(tree: DocNode) -> list[Section]:
    # Section is the compatibility DTO; this module and sections must not cycle.
    return [section_from_node(node) for node in content_blocks(tree)]


def section_from_node(node: DocNode) -> Section:
    path = node.hints.get("heading_path", ())
    return Section(
        " > ".join(path) or None,
        tree_text(node),
        PageContentType.TABLE if node.kind == NodeKind.TABLE else PageContentType.TEXT,
        dict(node.hints.get("metadata", {})),
    )


def tree_heading_context(tree: DocNode) -> dict[str, Any]:
    paths: list[tuple[str, ...]] = []

    def collect(node: DocNode, parents: tuple[str, ...]) -> None:
        # A heading without content blocks yields no chunk, so it is not a navigable scope.
        if node.kind == NodeKind.HEADING_BLOCK and not content_blocks(node):
            return

        path = (*parents, node.label) if node.kind == NodeKind.HEADING_BLOCK and node.label else parents

        if path and path not in paths:
            paths.append(path)

        for child in node.children:
            collect(child, path)

    collect(tree, ())
    keys = {path: " > ".join(path) for path in paths}
    groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}

    for path in paths:
        groups.setdefault(path[:-1], []).append(path)

    previous = {}
    following = {}
    siblings = {}

    for group in groups.values():
        for i, path in enumerate(group):
            previous[keys[path]] = group[i - 1][-1] if i else None
            following[keys[path]] = group[i + 1][-1] if i + 1 < len(group) else None
            siblings[keys[path]] = [other[-1] for other in group if other != path]

    depth = min(map(len, paths), default=0)
    return {
        "level": {keys[path]: len(path) for path in paths},
        "immediate": {keys[path]: path[-1] for path in paths},
        "parent": {keys[path]: " > ".join(path[:-1]) or None for path in paths},
        "prev": previous,
        "next": following,
        "siblings": siblings,
        "toc": list(dict.fromkeys(path[-1] for path in paths if len(path) == depth)),
    }
