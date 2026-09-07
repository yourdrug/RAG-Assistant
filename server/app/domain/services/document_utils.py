"""Document utility functions — storage key generation and filename resolution."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def generate_storage_key(owner_id: int | None, group_id: int | None, document_id: int, filename: str) -> str:
    """Build the storage key path for a document."""
    safe_name = Path(filename).name
    if owner_id is not None:
        return f"uploads/users/{owner_id}/{document_id}_{safe_name}"
    if group_id is not None:
        return f"uploads/groups/{group_id}/{document_id}_{safe_name}"
    return f"uploads/public/{document_id}_{safe_name}"


async def resolve_unique_filename(uow, owner_id: int | None, group_id: int | None, filename: str) -> str:
    """Find a unique filename by appending (N) suffix if needed."""
    p = Path(filename)
    stem = p.stem
    suffix = p.suffix
    candidate = filename
    counter = 1
    while await uow.documents.find_active_slot(owner_id, candidate, group_id) is not None:
        candidate = f"{stem}({counter}){suffix}"
        counter += 1
    if candidate != filename:
        log.info("Renamed conflict: %s -> %s", filename, candidate)
    return candidate
