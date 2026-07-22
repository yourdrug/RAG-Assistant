"""Document-related DTOs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentDTO:
    id: int
    filename: str
    visibility: str
    status: str
    error_message: str | None = None
    chunks: int | None = None
    chars: int | None = None


@dataclass(frozen=True)
class UploadDocumentCommand:
    filename: str
    file_data: bytes
    visibility: str
    group_id: int | None = None
    user_id: int | None = None
    user_kind: str = "internal"
    user_role: str = "user"
