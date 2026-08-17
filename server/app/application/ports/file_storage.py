"""FileStorage port — abstract interface for file storage operations.

Lives in the application layer so that application services depend on a port,
not on the concrete S3/local implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class FileItem:
    key: str
    filename: str
    size_bytes: int
    last_modified: str
    extension: str


@runtime_checkable
class FileStorage(Protocol):
    def list_files(self, prefix: str) -> list[FileItem]: ...
    def download_to_temp(self, key: str) -> Path: ...
    def upload_file(self, key: str, data: bytes) -> None: ...
    def get_file_info(self, key: str) -> FileItem | None: ...
    def delete_file(self, key: str) -> None: ...

    @property
    def supported_extensions(self) -> tuple[str, ...]: ...
