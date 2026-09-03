"""DocumentStatus value object -- lifecycle states for uploaded documents.

Valid transitions: PENDING -> PROCESSING -> INDEXING -> DONE | FAILED.
"""

from __future__ import annotations

from enum import StrEnum


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"
