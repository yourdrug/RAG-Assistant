"""Shared request context for ingestion workflows."""

from __future__ import annotations

from dataclasses import dataclass

from domain.knowledge.enums import DocumentVisibility, DomainDetectionMode


@dataclass(frozen=True, slots=True)
class IngestionScope:
    """Classification and ACL metadata applied to every produced chunk."""

    domain: str = DomainDetectionMode.AUTO.value
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC
    group_id: int | None = None
    client_id: int | None = None
