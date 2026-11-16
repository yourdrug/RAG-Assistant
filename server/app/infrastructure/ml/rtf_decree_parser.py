"""RTF decree parser — extraction of raw text and header attributes only.

RTF-specific responsibility ends at raw text + document-level header metadata
(decree number/date). Splitting the postanovlyayushchaya part into points/
subpoints is fully delegated to the shared content-based splitter with the
profile's content_boundaries() — no ad-hoc point-search logic here. The profile
(an OperativePartProfile from the active domain pack) knows the header requisites
and the operative-part marker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from striprtf.striprtf import rtf_to_text

from config import ChunkingSection, process_section
from domain.ingestion.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS, SplitUnit
from domain.ingestion.tree_packer import select_structural_units
from infrastructure.ml.tokenization import build_token_counter


if TYPE_CHECKING:
    from pathlib import Path

    from domain.ingestion.ports import DomainSettingsPort, TokenCounter
    from domain.ingestion.profiles.protocol import OperativePartProfile

_HEADER_ZONE_CHARS = 1500  # decree requisites are always at the top of the document


def parse_decree_rtf(
    file_path: Path,
    profile: OperativePartProfile,
    settings: DomainSettingsPort,
    token_counter: TokenCounter | None = None,
) -> tuple[list[SplitUnit], dict[str, str]]:
    """Parse a decree RTF into split units + document-level header metadata."""
    raw = rtf_to_text(file_path.read_text(encoding="utf-8", errors="replace"))

    doc_metadata = profile.header_requisites(raw[:_HEADER_ZONE_CHARS])
    split_span = profile.operative_part_span(raw)
    max_tokens = profile.chunking_policy(settings).max_tokens
    counter = token_counter or build_token_counter(process_section(ChunkingSection).embed_tokenizer_path)

    if split_span is None:
        # Decree-constatation without a postanovlyayushchaya part — boundaries
        # still apply: the coarsest level finds no matches and hands the text
        # to the next level, down to sentence if needed.
        return select_structural_units(
            raw,
            profile.content_boundaries(),
            max_tokens,
            min_chunk_chars=MIN_STRUCTURAL_CHUNK_CHARS,
            unit_label=profile.unit_label,
            measure=counter.count,
        ), doc_metadata

    preamble = raw[: split_span[0]].strip()
    body = raw[split_span[1] :]

    units: list[SplitUnit] = []

    if preamble:
        units.append(SplitUnit(None, preamble, "preamble"))

    units.extend(
        select_structural_units(
            body,
            profile.content_boundaries(),
            max_tokens,
            min_chunk_chars=MIN_STRUCTURAL_CHUNK_CHARS,
            unit_label=profile.unit_label,
            measure=counter.count,
        )
    )
    return units, doc_metadata
