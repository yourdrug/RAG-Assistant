"""Unified application facade for S3 document ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.ingestion.dto import (
    IngestRegistryItemDTO,
    IngestRegistryResult,
    IngestStatusResult,
)
from application.ingestion.scope import IngestionScope
from application.ingestion.targets import S3IngestionTargets
from domain.value_objects.visibility import DocumentVisibility


if TYPE_CHECKING:
    from collections.abc import Sequence

    from application.ingestion.batch import BatchIngestionWorkflow
    from application.ingestion.dto import (
        UploadFile,
    )
    from application.ingestion.registry import IngestionRegistry
    from application.ingestion.single_file import SingleFileIngestionWorkflow
    from application.ingestion.targets import S3UploadService


class IngestionService:
    """Public ingestion operations backed by workflows and storage collaborators."""

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

    async def run_full(
        self,
        docs_dir: str | None = None,
        *,
        reset: bool = False,
        domain: str = "auto",
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> IngestStatusResult:
        resolved = self._targets.resolve_docs_dir(docs_dir or "docs/")
        await self._batch.run(
            resolved,
            reset=reset,
            scope=IngestionScope(domain, visibility, group_id, client_id),
        )
        mode = "RESET + full reindex" if reset else "APPEND (new files only)"
        return IngestStatusResult(status="started", mode=mode, docs_dir=resolved)

    async def run_single(
        self,
        file_path: str,
        *,
        force: bool = False,
        domain: str = "auto",
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> IngestStatusResult:
        resolved = self._targets.resolve_ingest_target(file_path)

        if force:
            await self.force_reindex(resolved.rsplit("/", 1)[-1])

        await self._single_file.run(
            resolved,
            scope=IngestionScope(domain, visibility, group_id, client_id),
        )
        return IngestStatusResult(status="started", file=resolved, force=force)

    async def get_registry(self) -> IngestRegistryResult:
        raw = await self._registry.list_all()
        items = []

        for filename, info in raw.items():
            items.append(
                IngestRegistryItemDTO(
                    filename=filename,
                    chunks=info.get("chunks", 0),
                    chars=info.get("chars", 0),
                    indexed_at=info.get("indexed_at"),
                    source=info.get("source", ""),
                )
            )

        items.sort(key=lambda x: x.filename)
        total_chunks = sum(i.chunks for i in items)
        return IngestRegistryResult(
            total_files=len(items),
            total_chunks=total_chunks,
            files=items,
        )

    def resolve_docs_dir(self, prefix: str) -> str:
        return self._targets.resolve_docs_dir(prefix)

    def resolve_ingest_target(self, file_path: str) -> str:
        return self._targets.resolve_ingest_target(file_path)

    async def force_reindex(self, filename: str) -> None:
        await self._registry.delete(filename)

    async def upload_files(self, files: Sequence[UploadFile], prefix: str = "docs/") -> list[str]:
        return await self._uploads.upload_files(files, prefix)

    @staticmethod
    def validate_s3_key(key: str) -> None:
        S3IngestionTargets.validate_key(key)
