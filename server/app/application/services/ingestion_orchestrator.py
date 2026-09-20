"""Stable :class:`IngestionPort` facade for S3 ingestion.

Workflow and storage collaborators are constructed by the composition layer;
this class only adapts them to the public ingestion port.
"""

from __future__ import annotations

from typing import Any

from application.services.batch_ingestion import BatchIngestionWorkflow
from application.services.ingestion_registry import IngestionRegistry, s3_file_hash
from application.services.ingestion_scope import IngestionScope
from application.services.ingestion_targets import S3IngestionTargets, S3UploadService
from application.services.single_file_ingestion import SingleFileIngestionWorkflow
from domain.value_objects.visibility import DocumentVisibility

# Kept as an import-compatible alias for callers that used the old helper.
# TODO remove
_s3_file_hash = s3_file_hash


class IngestionService:
    """Port facade delegating work to injected ingestion collaborators."""

    def __init__(
        self,
        batch_workflow: BatchIngestionWorkflow,
        single_file_workflow: SingleFileIngestionWorkflow,
        registry: IngestionRegistry,
        targets: S3IngestionTargets,
        uploads: S3UploadService,
    ) -> None:
        self._batch = batch_workflow
        self._single_file = single_file_workflow
        self._registry = registry
        self._targets = targets
        self._uploads = uploads

    async def run_full_ingestion(
        self,
        docs_dir: str | None = None,
        reset: bool = False,
        domain: str = "auto",  # TODO replace
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None:
        await self._batch.run(
            docs_dir,
            reset=reset,
            scope=IngestionScope(domain, visibility, group_id, client_id),
        )

    async def run_single_file(
        self,
        file_path: str,
        domain: str = "auto",
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None:
        await self._single_file.run(
            file_path,
            scope=IngestionScope(domain, visibility, group_id, client_id),
        )

    async def upload_files(self, files: Any, prefix: str = "docs/") -> list[str]:
        return await self._uploads.upload_files(files, prefix)

    async def get_registry(self) -> dict:
        return await self._registry.list_all()

    async def force_reindex(self, filename: str) -> None:
        await self._registry.delete(filename)

    @staticmethod
    def _validate_s3_key(key: str) -> None:
        S3IngestionTargets.validate_key(key)

    def resolve_ingest_target(self, file_path: str) -> str:
        return self._targets.resolve_ingest_target(file_path)

    def resolve_docs_dir(self, docs_dir: str) -> str:
        return self._targets.resolve_docs_dir(docs_dir)
