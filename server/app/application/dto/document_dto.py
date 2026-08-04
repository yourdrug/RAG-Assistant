"""Document-related DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from domain.value_objects.roles import UserKind, UserRole


@dataclass(frozen=True)
class DocumentDTO:
    id: int
    filename: str
    visibility: str
    status: str
    source_path: str | None = None
    creation_date: datetime | None = None
    indexed_at: datetime | None = None
    error_message: str | None = None
    warning_message: str | None = None
    chunks: int | None = None
    chars: int | None = None
    storage_key: str | None = None
    replace_id: int | None = None
    owner_id: int | None = None
    group_id: int | None = None


@dataclass(frozen=True)
class UploadDocumentCommand:
    filename: str
    file_data: bytes
    visibility: str
    group_id: int | None = None
    user_id: int | None = None
    user_kind: str = UserKind.INTERNAL
    user_role: str = UserRole.USER
