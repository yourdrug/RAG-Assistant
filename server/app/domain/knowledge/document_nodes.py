"""Stable parent-node identities of chunks and their stored records (parent-child retrieval).

A node id is the document-local path of ancestor tokens: ``kind:number`` or ``kind:<label hash>``
joined by ``/``, with ``#N`` for the N-th separate occurrence of the same token under one parent.
It does not depend on character offsets, so re-splitting the same file yields the same ids.
Storage qualifies it with ``document_id`` and ``act_version_id``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from typing import TYPE_CHECKING, Any

from domain.shared.hashing import content_hash


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


PARENT_NODE_ID_KEY = "parent_node_id"
# Transient split metadata: ancestors with labels, consumed when nodes are persisted.
NODE_PATH_KEY = "node_path"
NODE_SEPARATOR = "/"
NODE_LABEL_MAX_CHARS = 500
# Parent units that give context to their followers but do not contain them.
CONTEXT_ONLY_UNIT_KINDS = frozenset({"preamble", "table_context"})
HEADING_NODE_KIND = "heading"
APPENDIX_NODE_KIND = "appendix"


@dataclass(frozen=True)
class NodeSegment:
    kind: str
    label: str
    number: str | None = None

    @property
    def token(self) -> str:
        if self.number:
            value = self.number.replace("%", "%25").replace(NODE_SEPARATOR, "%2F")
        else:
            value = hashlib.sha256(normalized_label(self.label).encode("utf-8")).hexdigest()[:10]

        return f"{self.kind}:{value}"


@dataclass(frozen=True)
class DocumentNode:
    """A parent scope of chunks; text is read from its chunks under the chunk ACL."""

    node_id: str
    parent_id: str | None
    kind: str
    label: str
    number: str | None
    depth: int
    chunk_start: int
    chunk_end: int
    pages: tuple[int, ...]
    char_count: int
    text_hash: str


def normalized_label(label: str) -> str:
    return " ".join(label.split()).casefold()


def node_segments(metadata: Mapping[str, Any], *, appendix: NodeSegment | None = None) -> list[NodeSegment]:
    """Ancestors of one chunk: appendix root, format headings, then structural parents."""
    candidates = [appendix] if appendix else []
    candidates.extend(
        NodeSegment(HEADING_NODE_KIND, heading) for heading in metadata.get("heading_path") or () if heading
    )
    candidates.extend(
        NodeSegment(
            str(parent.get("unit_kind") or HEADING_NODE_KIND),
            str(parent.get("heading") or ""),
            str(parent["boundary_value"]) if parent.get("boundary_value") else None,
        )
        for parent in metadata.get("parent_units") or ()
        if isinstance(parent, dict)
        and parent.get("unit_kind") not in CONTEXT_ONLY_UNIT_KINDS
        and (parent.get("heading") or parent.get("boundary_value"))
    )
    result: list[NodeSegment] = []

    for segment in candidates:
        # A style heading and the structural unit it introduces are one scope.
        if result and normalized_label(result[-1].label) == normalized_label(segment.label):
            continue

        result.append(segment)

    return result


def assign_node_paths(paths: Sequence[Sequence[NodeSegment]]) -> list[tuple[str, ...]]:
    """Resolve chunk ancestor paths in document order into unique node id prefixes.

    A token continues the open child of its parent; when it reappears after another child
    (or after text directly in the parent) it is a new occurrence and gets ``#N``.
    """
    open_child: dict[tuple[str, ...], tuple[str, str] | None] = {}
    seen: dict[tuple[str, ...], Counter[str]] = {}
    result = []

    for segments in paths:
        prefix: tuple[str, ...] = ()

        for segment in segments:
            raw = segment.token
            current = open_child.get(prefix)

            if current is not None and current[0] == raw:
                resolved = current[1]
            else:
                counter = seen.setdefault(prefix, Counter())
                counter[raw] += 1
                resolved = raw if counter[raw] == 1 else f"{raw}#{counter[raw]}"
                open_child[prefix] = (raw, resolved)

            prefix = (*prefix, resolved)

        open_child[prefix] = None
        result.append(prefix)

    return result


def assign_chunk_nodes(
    chunks: Sequence[tuple[Mapping[str, Any], NodeSegment | None]],
) -> list[dict[str, Any]]:
    """Return per-chunk ``parent_node_id`` and transient ``node_path`` metadata."""
    segments = [node_segments(metadata, appendix=appendix) for metadata, appendix in chunks]
    result = []

    for path, chunk_segments in zip(assign_node_paths(segments), segments, strict=True):
        node_path = [
            {
                "id": NODE_SEPARATOR.join(path[: depth + 1]),
                "kind": segment.kind,
                "label": segment.label[:NODE_LABEL_MAX_CHARS],
                "number": segment.number,
            }
            for depth, segment in enumerate(chunk_segments)
        ]
        result.append({PARENT_NODE_ID_KEY: NODE_SEPARATOR.join(path) or None, NODE_PATH_KEY: node_path})

    return result


def chunk_pages(metadata: Mapping[str, Any]) -> set[int]:
    pages = metadata.get("pages")

    if isinstance(pages, list):
        return {page for page in pages if isinstance(page, int)}

    start = metadata.get("page_start", metadata.get("page"))
    end = metadata.get("page_end", start)

    if isinstance(start, int) and isinstance(end, int) and start <= end:
        return set(range(start, end + 1))

    return set()


def build_document_nodes(chunks: Sequence[tuple[str, Mapping[str, Any]]]) -> list[DocumentNode]:
    """Aggregate ancestor scopes over persisted chunk order (index = chunk_index)."""
    nodes: dict[str, dict[str, Any]] = {}

    for index, (content, metadata) in enumerate(chunks):
        parent_id = None

        for depth, entry in enumerate(metadata.get(NODE_PATH_KEY) or ()):
            node = nodes.setdefault(
                entry["id"],
                {
                    "parent_id": parent_id,
                    "kind": entry["kind"],
                    "label": entry["label"],
                    "number": entry.get("number"),
                    "depth": depth + 1,
                    "start": index,
                    "end": index,
                    "pages": set(),
                    "texts": [],
                },
            )
            node["end"] = index
            node["pages"].update(chunk_pages(metadata))
            node["texts"].append(content)
            parent_id = entry["id"]

    return [
        DocumentNode(
            node_id=node_id,
            parent_id=node["parent_id"],
            kind=node["kind"],
            label=node["label"],
            number=node["number"],
            depth=node["depth"],
            chunk_start=node["start"],
            chunk_end=node["end"],
            pages=tuple(sorted(node["pages"])),
            char_count=sum(map(len, node["texts"])),
            text_hash=content_hash("\n".join(node["texts"])),
        )
        for node_id, node in nodes.items()
    ]


def node_ancestry(node_id: str) -> list[str]:
    """The node itself first, then each ancestor up to depth 1."""
    parts = node_id.split(NODE_SEPARATOR)
    return [NODE_SEPARATOR.join(parts[:depth]) for depth in range(len(parts), 0, -1)]


def select_parent(chain: Sequence[DocumentNode], max_chars: int) -> DocumentNode | None:
    """Climb from the nearest parent while the scope fits; ancestors only grow."""
    selected = None

    for node in chain:
        if node.char_count > max_chars:
            break

        selected = node

    return selected


def in_scope(node_id: str, parent_node_id: str | None) -> bool:
    """Whether a chunk with ``parent_node_id`` lies inside the node (or is its direct child)."""
    return bool(parent_node_id) and (
        parent_node_id == node_id or str(parent_node_id).startswith(node_id + NODE_SEPARATOR)
    )


def within_chunk_limit(chain: Sequence[DocumentNode], max_chunks: int) -> list[DocumentNode]:
    """Nearest ancestors whose whole chunk range can be read without truncation."""
    result = []

    for node in chain:
        if node.chunk_end - node.chunk_start + 1 > max_chunks:
            break

        result.append(node)

    return result


def measured_chain(
    chain: Sequence[DocumentNode], chunks: Sequence[tuple[str | None, str]]
) -> list[DocumentNode]:
    """Replace stored sizes with the current visible chunk bodies (manual edits change them)."""
    return [
        replace(
            node, char_count=sum(len(content) for parent, content in chunks if in_scope(node.node_id, parent))
        )
        for node in chain
    ]
