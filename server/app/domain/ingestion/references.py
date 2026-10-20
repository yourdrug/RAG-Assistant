"""Pure rules for regulatory-act identity, visibility and version intervals."""

from __future__ import annotations


from domain.ingestion.profiles.types import ReferenceMatch

_ACT_NUMBER_KINDS = ("decree_number", "act_number")


def extract_act_number(extracted_refs: list[ReferenceMatch]) -> str | None:
    for ref in extracted_refs:
        if ref.kind in _ACT_NUMBER_KINDS:
            return ref.value

    return None
