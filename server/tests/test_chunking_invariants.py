"""Chunking invariants shared by every chunking phase (docs/work/chunking-plan.md, 0.4).

The checks describe properties of good chunks, not their exact shape, so they survive
re-splitting. Violations the current splitter is known to have are marked strict xfail
with the plan item that fixes them: once fixed, the XPASS fails and the mark must go.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any

import pytest

from domain.ingestion.profiles.catalog import SETTINGS_PROFILE_TYPES
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.enums import DocumentVisibility, PageContentType
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.ingestion.splitting import settings_chunk_size
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from tests.chunking_fixtures import CHUNKING_FIXTURES, source_text


if TYPE_CHECKING:
    from pathlib import Path

    from tests.chunking_fixtures import ChunkingFixture


ACL_METADATA = {"owner_id": 42, "visibility": DocumentVisibility.CLIENT_PRIVATE.value, "group_id": 7}
INTERNAL_LEVEL_RE = re.compile(
    r"\b(chapter|section|article|clause|point|subpoint(?:_num)?|sentence|preamble|paragraph)\b"
)
STRUCTURAL_MARKER_RE = re.compile(
    r"^(глава|раздел|статья|приложение)\s+\S+|^\d+(\.\d+)*\.?$|^[а-яё]\)$", re.IGNORECASE
)
SENTENCE_END = (".", "!", "?", ";", ":")
# Whitespace and layout characters that differ between source and chunk text.
COMPACT_NOISE_RE = re.compile(r"[\s|]+")
SECTION_SEPARATOR_RE = re.compile(" > | \u203a ")


def compact(text: str) -> str:
    return COMPACT_NOISE_RE.sub("", text)


@dataclass(frozen=True)
class SplitResult:
    fixture: ChunkingFixture
    raw: list[RawDocument]
    chunks: list[RawDocument]
    max_chars: int


@pytest.fixture(scope="module")
def ingestion() -> tuple[LangchainDocumentParser, LangchainDocumentSplitter, DomainSettingsAdapter]:
    settings = DomainSettingsAdapter()
    values: dict[tuple[str, str], str] = {}

    for profile_type in SETTINGS_PROFILE_TYPES:
        profile = profile_type(settings=settings)

        for default in profile.config_defaults():
            values[default.key, profile.key] = default.value

    settings.replace_snapshot(values)
    registry = DomainProfileRegistry()
    register_all_profiles(registry, settings)
    return (
        LangchainDocumentParser(registry, settings),
        LangchainDocumentSplitter(registry, settings),
        settings,
    )


@pytest.fixture(scope="module")
def split_results(
    ingestion: tuple[LangchainDocumentParser, LangchainDocumentSplitter, DomainSettingsAdapter],
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, SplitResult]:
    parser, splitter, settings = ingestion
    results = {}

    for fixture in CHUNKING_FIXTURES:
        directory: Path = tmp_path_factory.mktemp(fixture.case_id)
        raw = parser.parse(fixture.render(directory))

        for doc in raw:
            doc.metadata.update(ACL_METADATA)

        chunks = splitter.split(
            [RawDocument(d.page_content, dict(d.metadata)) for d in raw], domain=fixture.domain
        )
        results[fixture.case_id] = SplitResult(
            fixture, raw, chunks, settings_chunk_size(fixture.domain, settings)
        )

    return results


def body(chunk: RawDocument) -> str:
    """Chunk text without the repeated structural context prefix."""
    prefix = chunk.metadata.get("context_prefix") or ""
    content = chunk.page_content
    return content[len(prefix) :] if prefix and content.startswith(prefix) else content


def is_table(chunk: RawDocument) -> bool:
    return chunk.metadata.get("content_type") == PageContentType.TABLE.value


def chunk_pages(metadata: dict[str, Any]) -> set[int]:
    pages = set(metadata.get("pages") or [])

    if metadata.get("page_start") is not None:
        end = metadata.get("page_end") or metadata["page_start"]
        pages.update(range(metadata["page_start"], end + 1))

    return pages


def page_index(raw: list[RawDocument]) -> tuple[str, list[tuple[int, int, int]]]:
    """Compact text of all text pages with (start, end, page) ranges."""
    text, spans = "", []

    for doc in raw:
        page = doc.metadata.get("page")
        part = compact(doc.page_content)

        if page is not None and not is_table(doc):
            spans.append((len(text), len(text) + len(part), int(page)))

        text += part

    return text, spans


def case(invariant: str, xfails: dict[str, str] | None = None) -> list[Any]:
    """Parametrize over fixtures; known violations become strict xfails."""
    xfails = xfails or {}
    return [
        pytest.param(
            fixture.case_id,
            id=fixture.case_id,
            marks=[pytest.mark.xfail(strict=True, reason=f"{invariant}: {xfails[fixture.case_id]}")]
            if fixture.case_id in xfails
            else [],
        )
        for fixture in CHUNKING_FIXTURES
    ]


def test_fixture_matrix_covers_every_planned_document_kind() -> None:
    names = {fixture.name for fixture in CHUNKING_FIXTURES}
    assert names == {"law", "decree", "contract", "instruction", "pdf_headings", "pdf_hard_wraps", "table"}
    assert all(fixture.source.is_file() for fixture in CHUNKING_FIXTURES)


@pytest.mark.parametrize("case_id", case("body is a source substring"))
def test_1_every_body_is_a_source_substring(case_id: str, split_results: dict[str, SplitResult]) -> None:
    result = split_results[case_id]
    source = compact("\n".join(doc.page_content for doc in result.raw))
    assert result.chunks

    for chunk in result.chunks:
        parts = body(chunk).splitlines() if is_table(chunk) else [body(chunk)]
        missing = [part for part in parts if compact(part) not in source]
        assert not missing, f"chunk {chunk.metadata.get('chunk_index')}: {missing[0][:120]!r}"


@pytest.mark.parametrize("case_id", case("meaningful text is covered"))
def test_2_chunks_cover_all_meaningful_source_text(
    case_id: str, split_results: dict[str, SplitResult]
) -> None:
    """Mark source characters found in some chunk; the context prefix and section path count."""
    result = split_results[case_id]
    source = compact(source_text(result.fixture))
    covered = bytearray(len(source))

    for chunk in result.chunks:
        section_path = re.split(SECTION_SEPARATOR_RE, chunk.metadata.get("section") or "")
        pieces = [body(chunk), *chunk.page_content.splitlines(), *section_path]
        pieces += [cell for line in pieces for cell in line.split("|")] if is_table(chunk) else []

        for piece in filter(None, map(compact, pieces)):
            start = source.find(piece)

            while start >= 0:
                covered[start : start + len(piece)] = b"\x01" * len(piece)
                start = source.find(piece, start + 1)

    uncovered = "".join("\x00" if flag else ch for ch, flag in zip(source, covered, strict=True))
    gaps = [gap for gap in re.findall(r"[^\x00]+", uncovered) if re.search(r"\w{3}", gap)]
    assert not gaps, f"{len(gaps)} source fragments lost, first: {gaps[0][:120]!r}"


@pytest.mark.parametrize("case_id", case("size within budget"))
def test_3_every_chunk_fits_the_budget(case_id: str, split_results: dict[str, SplitResult]) -> None:
    # Budget is in characters until the token counter (plan 3.1) replaces it.
    result = split_results[case_id]
    oversized = [len(c.page_content) for c in result.chunks if len(c.page_content) > result.max_chars]
    assert not oversized, f"limit {result.max_chars}, got {oversized}"


MAX_LEAD_IN_CHARS = 100


def is_heading_only(text: str, other_prefixes: str) -> bool:
    """Empty, repeated as another chunk's context, a bare marker, or a lone lead-in."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines or (compact(text) in other_prefixes and not text.endswith(SENTENCE_END)):
        return True

    bare_marker = (
        not text.endswith(SENTENCE_END)
        and len(lines) <= 2  # marker line plus its title
        and any(STRUCTURAL_MARKER_RE.match(line) for line in lines)
    )
    lone_lead_in = len(lines) == 1 and text.endswith(":") and len(text) <= MAX_LEAD_IN_CHARS
    return bare_marker or lone_lead_in


@pytest.mark.parametrize("case_id", case("no heading-only chunks"))
def test_4_no_chunk_is_only_a_heading_or_prefix(case_id: str, split_results: dict[str, SplitResult]) -> None:
    chunks = split_results[case_id].chunks
    offenders = [
        body(chunk).strip()[:80]
        for i, chunk in enumerate(chunks)
        if not is_table(chunk)
        and is_heading_only(
            body(chunk).strip(),
            "\x00".join(
                compact(c.metadata.get("context_prefix") or "") for j, c in enumerate(chunks) if j != i
            ),
        )
    ]
    assert not offenders, offenders


@pytest.mark.parametrize("case_id", case("pages match body"))
def test_5_pages_match_the_body(case_id: str, split_results: dict[str, SplitResult]) -> None:
    result = split_results[case_id]
    source, spans = page_index(result.raw)

    if not spans:
        pytest.skip("format without pages")

    cursor = 0

    for chunk in (c for c in result.chunks if not is_table(c)):
        text = compact(body(chunk))
        start = source.find(text, cursor)

        if start < 0:
            start = source.find(text)

        assert start >= 0, f"chunk {chunk.metadata.get('chunk_index')} not in source"
        end = start + len(text)
        cursor = start
        touched = {page for lo, hi, page in spans if lo < end and start < hi}
        assert chunk_pages(chunk.metadata) == touched, chunk.page_content[:120]


@pytest.mark.parametrize("case_id", case("ACL metadata propagated"))
def test_6_acl_metadata_reaches_every_chunk(case_id: str, split_results: dict[str, SplitResult]) -> None:
    for chunk in split_results[case_id].chunks:
        assert {key: chunk.metadata.get(key) for key in ACL_METADATA} == ACL_METADATA


@pytest.mark.parametrize("case_id", case("section is human readable"))
def test_7_section_has_no_internal_level_names(case_id: str, split_results: dict[str, SplitResult]) -> None:
    sections = {c.metadata.get("section") or "" for c in split_results[case_id].chunks}
    leaked = sorted(s for s in sections if INTERNAL_LEVEL_RE.search(s))
    assert not leaked, leaked
