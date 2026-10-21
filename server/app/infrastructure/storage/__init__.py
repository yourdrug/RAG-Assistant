"""S3-compatible file storage abstraction."""

from infrastructure.storage.file_storage import (
    FileItem,
    FileStorage,
    LazyStorage,
    S3Storage,
    create_storage,
)

__all__ = ["FileStorage", "FileItem", "LazyStorage", "S3Storage", "create_storage"]
