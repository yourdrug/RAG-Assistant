"""DocumentStatus value object -- lifecycle states for uploaded documents.

Valid transitions: PENDING -> PROCESSING -> INDEXING -> DONE | FAILED.
"""

from __future__ import annotations

from enum import StrEnum

from domain.value_objects.lifecycle_status import LifecycleStatusMixin


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
