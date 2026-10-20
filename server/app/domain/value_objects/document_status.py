"""DocumentStatus value object -- lifecycle states for uploaded documents.

Valid transitions: PENDING -> PROCESSING -> INDEXING -> DONE | FAILED.
"""

from __future__ import annotations

from enum import StrEnum

from domain.shared.enums import LifecycleStatusMixin


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


# Completed and failed documents retain their filename until explicitly removed.
DOCUMENT_SLOT_STATUSES = (
    DocumentStatus.PENDING,
    DocumentStatus.PROCESSING,
    DocumentStatus.INDEXING,
    DocumentStatus.DONE,
    DocumentStatus.FAILED,
)
