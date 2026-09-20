"""Storage-key validation and upload operations for S3 ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from application.ports.file_storage import FileStorage

log = logging.getLogger("default")


class S3IngestionTargets:
    """Validates object keys before storage operations use them."""

    @staticmethod
    def validate_key(key: str) -> None:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError("S3 key must not contain '..' or start with '/'")

    def resolve_ingest_target(self, file_path: str) -> str:
        self.validate_key(file_path)
        return file_path

    def resolve_docs_dir(self, docs_dir: str) -> str:
        self.validate_key(docs_dir)
        return docs_dir


class S3UploadService:
    """Uploads already-validated file content through the storage port."""

    def __init__(self, file_storage: "FileStorage", targets: S3IngestionTargets) -> None:
        self._file_storage = file_storage
        self._targets = targets

    async def upload_files(self, files: Any, prefix: str = "docs/") -> list[str]:
        self._targets.validate_key(prefix)
        uploaded: list[str] = []
        for file in files:
            key = f"{prefix}{file.filename}"
            self._targets.validate_key(key)
            await self._file_storage.upload_file(key, file.data)
            uploaded.append(key)
            log.info("Uploaded: %s (%d bytes)", key, len(file.data))
        return uploaded
