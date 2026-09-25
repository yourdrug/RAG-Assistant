"""RTF decree parser — extraction of raw text and header attributes only.

RTF-specific responsibility ends at raw text + document-level header metadata
(decree number/date). Splitting the postanovlyayushchaya part into points/
subpoints is fully delegated to the shared content-based splitter with
DecreeDomainProfile.content_boundaries() — no ad-hoc point-search logic here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from striprtf.striprtf import rtf_to_text

from domain.domain_profile.content_splitter import SplitUnit, split_by_content
from domain.domain_profile.profiles.decree import (
    _DECREE_DATE_RE,
    _DECREE_NUMBER_RE,
    _POSTANOVLYAYU_RE,
)

if TYPE_CHECKING:
    from domain.domain_profile.settings_port import DomainSettingsPort
    from domain.domain_profile.profiles.decree import DecreeDomainProfile

_HEADER_ZONE_CHARS = 1500  # decree requisites are always at the top of the document


def parse_decree_rtf(
    file_path: Path,
    profile: "DecreeDomainProfile",
    settings: "DomainSettingsPort",
) -> tuple[list[SplitUnit], dict]:
    """Parse a decree RTF into split units + document-level header metadata."""
    raw = rtf_to_text(file_path.read_text(encoding="utf-8", errors="replace"))

    header_zone = raw[:_HEADER_ZONE_CHARS]
    doc_metadata: dict = {}
    if m := _DECREE_NUMBER_RE.search(header_zone):
        doc_metadata["decree_number"] = m.group(1)
    if m := _DECREE_DATE_RE.search(header_zone):
        doc_metadata["decree_date"] = m.group(1)

    split_match = _POSTANOVLYAYU_RE.search(raw)
    max_chars = int(settings.get("max_unit_chars", domain_key=profile.key))
    if split_match is None:
        # Decree-constatation without a postanovlyayushchaya part — boundaries
        # still apply: the coarsest level finds no matches and hands the text
        # to the next level, down to sentence if needed.
        return split_by_content(raw, profile.content_boundaries(), max_chars), doc_metadata

    preamble = raw[: split_match.start()].strip()
    body = raw[split_match.end() :]

    units: list[SplitUnit] = []
    if preamble:
        units.append(SplitUnit(None, preamble, "preamble"))
    units.extend(split_by_content(body, profile.content_boundaries(), max_chars))
    return units, doc_metadata
