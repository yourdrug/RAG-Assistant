"""File input contract for ingestion uploads."""

from typing import Protocol


class UploadFile(Protocol):
    """File content accepted by ingestion upload operations."""

    @property
    def filename(self) -> str: ...

    @property
    def data(self) -> bytes: ...
