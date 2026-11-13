"""Sentence-aware source splitting with a measured word/character fallback."""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

from domain.ingestion.content_splitter import TINY_FRAGMENT_MAX_CHARS
from domain.ingestion.profiles.patterns import SENTENCE_RE


if TYPE_CHECKING:
    from collections.abc import Callable


def split_at_boundaries(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    """Slice at match offsets, retaining complete markers and capture groups."""
    offsets = sorted({0, len(text), *(m.start() for pattern in patterns for m in pattern.finditer(text))})
    return [text[start:end] for start, end in itertools.pairwise(offsets) if text[start:end].strip()]


def sentence_fragments(text: str) -> list[str]:
    """Keep exact source spans and the legal abbreviation dictionary."""
    offsets = [0, *(match.end() for match in SENTENCE_RE.finditer(text)), len(text)]
    return [text[start:end] for start, end in itertools.pairwise(offsets) if text[start:end].strip()]


def overlap_tail(text: str, size: int, measure: Callable[[str], int] = len) -> str:
    """Return complete trailing sentences whose total size fits the overlap."""
    tail = ""

    for fragment in reversed(sentence_fragments(text)):
        candidate = fragment + tail

        if measure(candidate) > size:
            break

        tail = candidate

    return tail if size > 0 else ""


def pack_sentences(
    fragments: list[str], chunk_size: int, chunk_overlap: int, measure: Callable[[str], int] = len
) -> list[str]:
    """Pack exact contiguous fragments, overlapping only complete sentences."""
    merged: list[str] = []
    current = ""

    for fragment in fragments:
        if current and measure(current + fragment) > chunk_size:
            merged.append(current.strip())
            current = overlap_tail(current, chunk_overlap, measure)

            if measure(current + fragment) > chunk_size:
                current = ""

        current += fragment

    if current.strip():
        merged.append(current.strip())

    return merged


def fallback_split(
    chunks: list[str],
    chunk_size: int,
    chunk_overlap: int,
    measure: Callable[[str], int] = len,
    *,
    on_fallback: Callable[[], None] | None = None,
) -> list[str]:
    """Split whole sentences first; only oversized sentences reach word/character splitting."""
    result = []

    for chunk in chunks:
        pending: list[str] = []

        for sentence in sentence_fragments(chunk):
            if measure(sentence) <= chunk_size:
                pending.append(sentence)
                continue

            result.extend(pack_sentences(pending, chunk_size, chunk_overlap, measure))
            pending = []

            if on_fallback:
                on_fallback()

            result.extend(split_words(sentence, chunk_size, measure))

        result.extend(pack_sentences(pending, chunk_size, chunk_overlap, measure))

    return result


def split_leaf_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    boundary_patterns: list[re.Pattern[str]] | None = None,
    measure: Callable[[str], int] = len,
    *,
    on_fallback: Callable[[], None] | None = None,
) -> list[str]:
    """Retain structural markers while enforcing a strict character budget."""
    if measure(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    if boundary_patterns:
        fragments = split_at_boundaries(text, boundary_patterns)
        chunks = pack_sentences(fragments, chunk_size, chunk_overlap, measure)
    else:
        chunks = [text]

    parts = fallback_split(chunks, chunk_size, chunk_overlap, measure, on_fallback=on_fallback)
    result = []
    index = 0

    while index < len(parts):
        part = parts[index]

        if (
            len(part) < TINY_FRAGMENT_MAX_CHARS
            and index + 1 < len(parts)
            and re.fullmatch(r"(?:\d+[.\d]*|Статья\s+\d+[.\d]*)", part)
        ):
            if parts[index + 1].startswith(part):
                # The next body already carries this marker through overlap.
                index += 1
                continue

            prefix = part + "\n"
            budget = chunk_size - measure(prefix)

            if budget > 0:
                following = fallback_split(
                    [parts[index + 1]],
                    budget,
                    min(chunk_overlap, budget - 1),
                    measure,
                    on_fallback=on_fallback,
                )
                result.extend([prefix + following[0], *following[1:]])
                index += 2
            else:
                result.append(part)
                index += 1
        else:
            result.append(part)
            index += 1

    return result


def pack_fragments(fragments: list[str], budget: int, measure: Callable[[str], int]) -> list[str]:
    """Preserve the historical measured join, including tokenizer special tokens."""
    result = []
    pending = ""
    total = 0
    separator_size = measure("")

    for fragment in fragments:
        size = measure(fragment)

        if size > budget:
            message = "A source character exceeds the token budget"
            raise ValueError(message)

        extra = separator_size if pending else 0

        if pending and total + size + extra > budget:
            if pending.strip():
                result.append(pending.strip())

            pending, total, extra = "", 0, 0

        pending += fragment
        total += size + extra

    if pending.strip():
        result.append(pending.strip())

    return result


def split_words(text: str, budget: int, measure: Callable[[str], int]) -> list[str]:
    """Pack whitespace-terminated words; descend to characters only for long words."""
    result = []
    pending = []

    for word in re.findall(r"\S*\s+|\S+$", text):
        if measure(word) < budget:
            pending.append(word)
            continue

        result.extend(pack_fragments(pending, budget, measure))
        pending = []
        result.extend(pack_fragments(list(word), budget, measure))

    result.extend(pack_fragments(pending, budget, measure))
    return result


def locate_span(source: str, content: str, cursor: int = 0) -> tuple[int, int] | None:
    """Locate text even if structural merging changed only whitespace."""
    start = source.find(content, cursor)

    if start >= 0:
        return start, start + len(content)

    pattern = r"\s+".join(re.escape(word) for word in content.split())

    if not pattern:
        return None

    match = re.search(pattern, source[cursor:])
    return (cursor + match.start(), cursor + match.end()) if match else None
