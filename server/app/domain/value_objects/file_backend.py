"""File storage backend type."""

from __future__ import annotations

from enum import StrEnum


class FileBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"
