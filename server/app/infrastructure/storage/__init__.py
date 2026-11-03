"""S3-compatible file storage abstraction."""

from application.ports.file_storage import FileItem, FileStorage
from infrastructure.storage.file_storage import (
    LazyStorage,
    S3Storage,
    create_storage,
)


__all__ = ["FileItem", "FileStorage", "LazyStorage", "S3Storage", "create_storage"]
