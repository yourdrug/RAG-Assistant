"""Build bounded dense context independently of storage and ML frameworks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from domain.ingestion.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS
from domain.knowledge.chunk_context import content_with_scope
from domain.shared.breadcrumb import shorten_breadcrumb


if TYPE_CHECKING:
    from collections.abc import Callable


MAX_EMBED_TOKENS = 2048
MAX_CONTEXT_CHARS = 300
CHARS_PER_TOKEN = 4
# Metadata key holding the exact prefix chosen at splitting; manual edits reuse it.
EMBED_PREFIX_KEY = "embed_prefix"


def estimate_tokens(text: str) -> int:
    """Character estimate used only when no embedding tokenizer is available."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def shorten_context(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text

    return text[: max(0, limit - 1)].rstrip() + "…"


def parent_introduction(metadata: dict[str, Any], body: str) -> str:
    """Only the nearest parent's substantive introduction accompanies the path."""
    parents = metadata.get("parent_units") or []

    if not parents:
        return ""

    parent = parents[-1]
    content: str = parent.get("content", "").strip()
    heading = " ".join(parent.get("heading", "").split())
    normalized = " ".join(content.split())

    if not normalized or normalized == heading or normalized in " ".join(body.split()):
        return ""

    # Enclosing headings can occupy two source lines; strip them as a block.
    lines = content.splitlines()

    for count in range(1, min(len(lines), 2) + 1):
        if " ".join(" ".join(lines[:count]).split()) == heading:
            return "\n".join(lines[count:]).strip()

    return content


def character_context_prefix(
    metadata: dict[str, Any], chunk_size: int, *, include_unit: bool = False, body: str = ""
) -> str:
    """Bound title and path, giving the nearest parent priority over distant ones."""
    remaining = chunk_size - min(MIN_STRUCTURAL_CHUNK_CHARS, chunk_size // 2)
    intro = parent_introduction(metadata, body)

    # Conditions are atomic: the complete version also remains in parent_units.
    if len(intro) + 2 > remaining:
        intro = ""

    remaining -= len(intro) + (2 if intro else 0)
    full_title = " ".join((metadata.get("doc_title") or "").split())
    title = shorten_context(full_title, 80)
    normalized_body = " ".join(body.split())

    if title and (normalized_body.startswith(full_title) or len(title) + 2 > remaining):
        title = ""

    remaining -= len(title) + (1 if title else 0)
    headings = (metadata.get("section") or "").split(" > ")
    heading = metadata.get("unit_heading_text") or ""

    if include_unit and heading and heading not in headings:
        headings.append(heading)

    headings = [part for part in headings if part and part not in (title, full_title)]

    # A heading that starts the actual body needs no second copy in the prefix.
    while headings and normalized_body.startswith(" ".join(headings[-1].split())):
        headings.pop()

    path_budget = min(120, chunk_size // 3, remaining - 2)
    path = shorten_breadcrumb(" > ".join(headings), path_budget) if headings and path_budget > 0 else ""
    parts = [part for part in (title, path, intro) if part]
    return "\n".join(parts) + "\n\n" if parts else ""


def context_prefix(
    metadata: dict[str, Any],
    chunk_size: int,
    *,
    include_unit: bool = False,
    body: str = "",
    measure: Callable[[str], int] = len,
) -> str:
    """Reserve at least half the measured budget for source text."""
    # Character bounding still limits title/path lengths, independently of token units.
    context_size = chunk_size if measure is len else chunk_size * CHARS_PER_TOKEN
    prefix = character_context_prefix(metadata, context_size, include_unit=include_unit, body=body)

    if measure(prefix) <= chunk_size // 2:
        return prefix

    # Introductory conditions stay whole in metadata; never truncate them mid-sentence.
    reduced = {**metadata, "parent_units": []}
    prefix = character_context_prefix(reduced, context_size, include_unit=include_unit, body=body)

    if measure(prefix) > chunk_size // 2:
        reduced["doc_title"] = ""
        prefix = character_context_prefix(reduced, context_size, include_unit=include_unit, body=body)

    return prefix if measure(prefix) <= chunk_size // 2 else ""


def build_embed_text(
    metadata: dict[str, Any],
    content: str,
    *,
    max_tokens: int = MAX_EMBED_TOKENS,
    measure: Callable[[str], int] = estimate_tokens,
) -> str:
    """Keep source text intact and add only context fitting the measured limit."""
    count = measure
    prefix = metadata.get(EMBED_PREFIX_KEY)

    if prefix is None:
        prefix = context_prefix(metadata, max_tokens, include_unit=True, body=content, measure=count)

    scoped = content_with_scope(metadata, prefix + content)

    if count(scoped) <= max_tokens:
        return scoped

    if count(prefix + content) > max_tokens:
        prefix = context_prefix(
            {**metadata, "doc_title": ""},
            max_tokens,
            include_unit=True,
            body=content,
            measure=count,
        )

    return prefix + content if count(prefix + content) <= max_tokens else content


def with_context_summary(
    embed_text: str,
    summary: str,
    *,
    max_tokens: int = MAX_EMBED_TOKENS,
    measure: Callable[[str], int] = estimate_tokens,
) -> str | None:
    """Prepend a generated situating note; ``None`` when it does not fit the dense limit."""
    note = " ".join(summary.split())

    if not note:
        return None

    combined = note + "\n\n" + embed_text
    return combined if measure(combined) <= max_tokens else None
