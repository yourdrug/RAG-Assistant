"""Shared request context for ingestion workflows."""

from __future__ import annotations

from dataclasses import dataclass

from domain.value_objects.doc_domain import DomainDetectionMode
from domain.value_objects.visibility import DocumentVisibility


@dataclass(frozen=True, slots=True)
class IngestionScope:
    """Classification and ACL metadata applied to every produced chunk."""

    domain: str = DomainDetectionMode.AUTO.value
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC
    group_id: int | None = None
    client_id: int | None = None
