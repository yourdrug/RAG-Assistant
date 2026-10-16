"""FileStorage port — abstract interface for file storage operations.

Lives in the application layer so that application services depend on a port,
not on the concrete S3/local implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@dataclass
class FileItem:
    key: str
    filename: str
    size_bytes: int
    last_modified: str
    extension: str


@runtime_checkable
class FileStorage(Protocol):
    async def list_files(self, prefix: str) -> list[FileItem]: ...

    async def download_to_temp(self, key: str) -> Path: ...

    async def upload_file(self, key: str, data: bytes) -> None: ...

    async def upload_stream(self, key: str, blocks: AsyncIterator[bytes]) -> None: ...

    async def get_file_info(self, key: str) -> FileItem | None: ...

    async def delete_file(self, key: str) -> None: ...

    async def rename_file(self, old_key: str, new_key: str) -> None: ...

    async def copy_file(self, old_key: str, new_key: str) -> None: ...

    async def aclose(self) -> None: ...

    @property
    def supported_extensions(self) -> tuple[str, ...]: ...
