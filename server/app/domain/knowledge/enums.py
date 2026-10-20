"""Knowledge domain: enums."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from domain.shared.enums import LifecycleStatusMixin, ValidatedEnumMixin


class DocDomain(StrEnum):
    GENERAL = "general"
    LEGAL = "legal"
    DECREE = "decree"


class DomainDetectionMode(StrEnum):
    AUTO = "auto"


DocDomainSelection = Literal[DocDomain.GENERAL, DocDomain.LEGAL, DocDomain.DECREE, DomainDetectionMode.AUTO]


class DocumentStatus(
    LifecycleStatusMixin,
    StrEnum,
    terminal=frozenset({"done", "failed"}),
    active=frozenset({"pending", "processing", "indexing"}),
):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"


DOCUMENT_SLOT_STATUSES = (
    DocumentStatus.PENDING,
    DocumentStatus.PROCESSING,
    DocumentStatus.INDEXING,
    DocumentStatus.DONE,
    DocumentStatus.FAILED,
)


class DocumentVisibility(ValidatedEnumMixin, StrEnum, label="visibility"):
    INTERNAL_PUBLIC = "internal_public"
    INTERNAL_GROUP = "internal_group"
    INTERNAL_PRIVATE = "internal_private"
    CLIENT_PRIVATE = "client_private"


class SourceType(StrEnum):
    FILE = "file"
    MANUAL = "manual"


class SearchMode(StrEnum):
    EXACT = "exact"
    ICONTAINS = "icontains"


class PageContentType(StrEnum):
    TEXT = "text"
    SCAN = "scan"
    GARBLED = "garbled"
    EMPTY = "empty"
    TABLE = "table"
    OCR = "ocr"
    IMAGE_ONLY = "image_only"

    @classmethod
    def empty_counts(cls) -> dict[str, int]:
        """Return a zero-initialized counter for all content types."""
        return {m.value: 0 for m in cls}
