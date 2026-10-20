"""Storage-key validation and upload operations for S3 ingestion."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4


if TYPE_CHECKING:
    from collections.abc import Sequence

    from application.ingestion.ports.upload import UploadFile
    from application.ports.file_storage import FileStorage

log = logging.getLogger("default")


class S3IngestionTargets:
    """Validates object keys before storage operations use them."""

    @staticmethod
    def validate_key(key: str) -> None:
        if key.startswith("/") or ".." in Path(key).parts:
            msg = "S3 key must not contain '..' or start with '/'"
            raise ValueError(msg)

    def resolve_ingest_target(self, file_path: str) -> str:
        self.validate_key(file_path)
        return file_path

    def resolve_docs_dir(self, docs_dir: str) -> str:
        self.validate_key(docs_dir)
        return docs_dir.rstrip("/") + "/"


class S3UploadService:
    """Uploads already-validated file content through the storage port."""

    def __init__(self, file_storage: FileStorage, targets: S3IngestionTargets) -> None:
        self._file_storage = file_storage
        self._targets = targets

    async def upload_files(self, files: Sequence[UploadFile], prefix: str = "docs/") -> list[str]:
        self._targets.validate_key(prefix)
        keys = [f"{prefix}{file.filename}" for file in files]

        for key in keys:
            self._targets.validate_key(key)

        backups: dict[str, str | None] = {}
        uploaded: list[str] = []
        attempted: list[str] = []
        try:
            for key in dict.fromkeys(keys):
                backups[key] = None

                if await self._file_storage.get_file_info(key) is not None:
                    # Extensionless backups stay outside document ingestion prefixes.
                    backup = f".upload-backups/{uuid4().hex}"
                    backups[key] = backup
                    await self._file_storage.copy_file(key, backup)

            for key, file in zip(keys, files, strict=True):
                # A failed upload may have reached storage before its response was lost.
                attempted.append(key)
                await self._file_storage.upload_file(key, file.data)
                uploaded.append(key)
                log.info("Uploaded: %s (%d bytes)", key, len(file.data))
        except (Exception, asyncio.CancelledError):
            await self.rollback_uploads(attempted, backups)
            raise
        finally:
            await self.cleanup_backups(backups)

        return uploaded

    async def rollback_uploads(self, keys: list[str], backups: dict[str, str | None]) -> None:
        for key in dict.fromkeys(reversed(keys)):
            backup = backups[key]
            try:
                if backup is None:
                    await self._file_storage.delete_file(key)
                else:
                    await self._file_storage.copy_file(backup, key)
            except Exception:
                # Retain the original object for recovery if restoration fails.
                backups.pop(key)
                log.exception("Failed to roll back batch upload: %s (backup: %s)", key, backup)

    async def cleanup_backups(self, backups: dict[str, str | None]) -> None:
        for backup in backups.values():
            if backup is not None:
                try:
                    await self._file_storage.delete_file(backup)
                except Exception:
                    log.exception("Failed to delete upload backup: %s", backup)
