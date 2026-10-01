"""Pure rules for regulatory-act identity, visibility and version intervals."""

from __future__ import annotations

from datetime import date

from domain.domain_profile.protocol import ReferenceMatch
from domain.entities.act_version import ActVersion
from domain.value_objects.visibility import DocumentVisibility

_ACT_NUMBER_KINDS = ("decree_number", "act_number")


def _extract_act_number(extracted_refs: list[ReferenceMatch]) -> str | None:
    for ref in extracted_refs:
        if ref.kind in _ACT_NUMBER_KINDS:
            return ref.value
    return None


def _same_document_scope(left, right) -> bool:
    if left.visibility != right.visibility:
        return False
    if left.visibility in (DocumentVisibility.INTERNAL_PRIVATE, DocumentVisibility.CLIENT_PRIVATE):
        return left.owner_id == right.owner_id
    if left.visibility == DocumentVisibility.INTERNAL_GROUP:
        return left.group_id == right.group_id
    return left.visibility == DocumentVisibility.INTERNAL_PUBLIC


def _document_scope_key(document) -> str:
    if document.visibility == DocumentVisibility.INTERNAL_GROUP:
        return f"{document.visibility}:{document.group_id}"
    if document.visibility in (DocumentVisibility.INTERNAL_PRIVATE, DocumentVisibility.CLIENT_PRIVATE):
        return f"{document.visibility}:{document.owner_id}"
    return str(document.visibility)


def _is_latest_version(previous: list[ActVersion], effective_date: date | None, *, today: date) -> bool:
    if effective_date is None:
        return not any(v.is_current for v in previous)
    if effective_date > today:
        return False
    current_dates = [
        v.effective_from for v in previous if v.effective_from is not None and v.effective_from <= today
    ]
    return not current_dates or effective_date >= max(current_dates)


def _new_version_effective_to(previous: list[ActVersion], effective_date: date | None) -> date | None:
    if effective_date is None:
        return None
    later_dates = [
        v.effective_from for v in previous if v.effective_from and v.effective_from > effective_date
    ]
    return min(later_dates) if later_dates else None
