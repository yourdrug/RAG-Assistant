"""Translate domain structural units into shared parser/splitter metadata."""

from dataclasses import asdict

from domain.domain_profile.content_splitter import SplitUnit
from domain.domain_profile.protocol import DomainProfile, refs_to_metadata
from infrastructure.ml.ingestion.text_chunks import shorten_context


def unit_metadata(unit: SplitUnit, profile: DomainProfile, metadata: dict) -> dict:
    """Keep parent identity, breadcrumb and scope on every descendant."""
    meta = {**metadata, "unit_kind": unit.unit_kind}
    parents = [asdict(parent) for parent in unit.parents]
    if parents:
        meta["parent_units"] = parents
    headings = [metadata["section"]] if metadata.get("section") else []
    for parent in unit.parents:
        if parent.heading not in headings:
            headings.append(parent.heading)
        if parent.boundary_value:
            meta[f"{parent.unit_kind}_number"] = parent.boundary_value
    if unit.heading and unit.heading not in headings:
        headings.append(unit.heading)
    if unit.heading:
        meta["unit_heading_text"] = shorten_context(unit.content.split("\n", 1)[0].strip())
    if headings:
        meta["section"] = " > ".join(headings)
    if unit.boundary_value:
        meta[f"{unit.unit_kind}_number"] = unit.boundary_value
    refs = [ref for parent in unit.parents for ref in profile.extract_references(parent.content)]
    refs.extend(profile.extract_references(unit.content))
    if refs:
        meta["domain_metadata"] = {**(metadata.get("domain_metadata") or {}), **refs_to_metadata(refs)}
    return meta
