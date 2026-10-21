"""Knowledge domain: policy."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from domain.knowledge.document import Document

from domain.knowledge.enums import DocumentVisibility


def same_document_scope(left: Document, right: Document) -> bool:
    if left.visibility != right.visibility:
        return False

    if left.visibility in (DocumentVisibility.INTERNAL_PRIVATE, DocumentVisibility.CLIENT_PRIVATE):
        return left.owner_id == right.owner_id

    if left.visibility == DocumentVisibility.INTERNAL_GROUP:
        return left.group_id == right.group_id

    return left.visibility == DocumentVisibility.INTERNAL_PUBLIC


def document_scope_key(document: Document) -> str:
    if document.visibility == DocumentVisibility.INTERNAL_GROUP:
        return f"{document.visibility}:{document.group_id}"

    if document.visibility in (DocumentVisibility.INTERNAL_PRIVATE, DocumentVisibility.CLIENT_PRIVATE):
        return f"{document.visibility}:{document.owner_id}"

    return str(document.visibility)
