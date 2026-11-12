"""Translate domain structural units into shared parser/splitter metadata."""

from dataclasses import asdict
from typing import Any

from domain.ingestion.content_splitter import SplitUnit
from domain.ingestion.profiles.protocol import DocumentProfile
from domain.ingestion.profiles.types import refs_to_metadata
from domain.knowledge.chunk import SENTENCE_UNIT_KIND


def unit_metadata(unit: SplitUnit, profile: DocumentProfile, metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep parent identity, breadcrumb and scope on every descendant."""
    meta = {**metadata, "unit_kind": unit.unit_kind}
    parents = [asdict(parent) for parent in unit.parents]

    if parents:
        meta["parent_units"] = parents

    populate_unit_headings(unit, metadata, meta)

    refs = [ref for parent in unit.parents for ref in profile.extract_references(parent.content)]
    refs.extend(profile.extract_references(unit.content))

    if refs:
        meta["domain_metadata"] = {**(metadata.get("domain_metadata") or {}), **refs_to_metadata(refs)}

    return meta


def populate_unit_headings(unit: SplitUnit, metadata: dict[str, Any], meta: dict[str, Any]) -> None:
    headings = [metadata["section"]] if metadata.get("section") else []

    for parent in unit.parents:
        if parent.heading not in headings:
            headings.append(parent.heading)

        if parent.boundary_value:
            meta[f"{parent.unit_kind}_number"] = parent.boundary_value

    if unit.unit_kind != SENTENCE_UNIT_KIND and unit.heading and unit.heading not in headings:
        headings.append(unit.heading)

    if unit.heading and unit.unit_kind != SENTENCE_UNIT_KIND:
        meta["unit_heading_text"] = unit.heading

    if headings:
        meta["section"] = " > ".join(headings)

    if unit.boundary_value:
        meta[f"{unit.unit_kind}_number"] = unit.boundary_value
