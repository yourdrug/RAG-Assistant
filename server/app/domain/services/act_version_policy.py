"""Pure rules for regulatory-act identity, visibility and version intervals."""

from __future__ import annotations


from domain.domain_profile.protocol import ReferenceMatch
from domain.value_objects.visibility import DocumentVisibility

_ACT_NUMBER_KINDS = ("decree_number", "act_number")


def extract_act_number(extracted_refs: list[ReferenceMatch]) -> str | None:
    for ref in extracted_refs:
        if ref.kind in _ACT_NUMBER_KINDS:
            return ref.value
    return None


def same_document_scope(left, right) -> bool:
    if left.visibility != right.visibility:
        return False
    if left.visibility in (DocumentVisibility.INTERNAL_PRIVATE, DocumentVisibility.CLIENT_PRIVATE):
        return left.owner_id == right.owner_id
    if left.visibility == DocumentVisibility.INTERNAL_GROUP:
        return left.group_id == right.group_id
    return left.visibility == DocumentVisibility.INTERNAL_PUBLIC


def document_scope_key(document) -> str:
    if document.visibility == DocumentVisibility.INTERNAL_GROUP:
        return f"{document.visibility}:{document.group_id}"
    if document.visibility in (DocumentVisibility.INTERNAL_PRIVATE, DocumentVisibility.CLIENT_PRIVATE):
        return f"{document.visibility}:{document.owner_id}"
    return str(document.visibility)
