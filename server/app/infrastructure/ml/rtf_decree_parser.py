"""RTF decree parser — extraction of raw text and header attributes only.

RTF-specific responsibility ends at raw text + document-level header metadata
(decree number/date). Splitting the postanovlyayushchaya part into points/
subpoints is fully delegated to the shared content-based splitter with
DecreeDomainProfile.content_boundaries() — no ad-hoc point-search logic here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from striprtf.striprtf import rtf_to_text

from config import ChunkingSection, process_section
from domain.ingestion.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS, SplitUnit, split_by_content
from domain.ingestion.profiles.decree import (
    _DECREE_DATE_RE,
    _DECREE_NUMBER_RE,
    _POSTANOVLYAYU_RE,
)
from infrastructure.ml.tokenization import build_token_counter


if TYPE_CHECKING:
    from pathlib import Path

    from domain.ingestion.ports import DomainSettingsPort, TokenCounter
    from domain.ingestion.profiles.protocol import DocumentProfile

_HEADER_ZONE_CHARS = 1500  # decree requisites are always at the top of the document


def parse_decree_rtf(
    file_path: Path,
    profile: DocumentProfile,
    settings: DomainSettingsPort,
    token_counter: TokenCounter | None = None,
) -> tuple[list[SplitUnit], dict[str, Any]]:
    """Parse a decree RTF into split units + document-level header metadata."""
    raw = rtf_to_text(file_path.read_text(encoding="utf-8", errors="replace"))

    header_zone = raw[:_HEADER_ZONE_CHARS]
    doc_metadata: dict[str, Any] = {}

    if m := _DECREE_NUMBER_RE.search(header_zone):
        doc_metadata["decree_number"] = m.group(1)

    if m := _DECREE_DATE_RE.search(header_zone):
        doc_metadata["decree_date"] = m.group(1)

    split_match = _POSTANOVLYAYU_RE.search(raw)
    max_tokens = int(settings.get(f"{profile.key}_chunk_max_tokens", domain_key=profile.key))
    counter = token_counter or build_token_counter(process_section(ChunkingSection).embed_tokenizer_path)

    if split_match is None:
        # Decree-constatation without a postanovlyayushchaya part — boundaries
        # still apply: the coarsest level finds no matches and hands the text
        # to the next level, down to sentence if needed.
        return split_by_content(
            raw,
            profile.content_boundaries(),
            max_tokens,
            min_chunk_chars=MIN_STRUCTURAL_CHUNK_CHARS,
            unit_label=profile.unit_label,
            measure=counter.count,
        ), doc_metadata

    preamble = raw[: split_match.start()].strip()
    body = raw[split_match.end() :]

    units: list[SplitUnit] = []

    if preamble:
        units.append(SplitUnit(None, preamble, "preamble"))

    units.extend(
        split_by_content(
            body,
            profile.content_boundaries(),
            max_tokens,
            min_chunk_chars=MIN_STRUCTURAL_CHUNK_CHARS,
            unit_label=profile.unit_label,
            measure=counter.count,
        )
    )
    return units, doc_metadata
