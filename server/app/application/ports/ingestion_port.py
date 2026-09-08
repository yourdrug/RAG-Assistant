"""IngestionPort -- application-layer abstraction for the S3-only document ingestion pipeline.

Defines the protocol that ``IngestAppService`` depends on.  The concrete
implementation lives in ``infrastructure.services`` and uses S3 as the
sole file storage backend.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from domain.value_objects.visibility import DocumentVisibility


@runtime_checkable
class IngestionPort(Protocol):
    def resolve_docs_dir(self, docs_dir: str) -> str: ...
    async def run_full_ingestion(
        self,
        docs_dir: str | None = None,
        reset: bool = False,
        domain: str = "auto",
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None: ...
    def resolve_ingest_target(self, file_path: str) -> str: ...
    async def run_single_file(
        self,
        file_path: str,
        domain: str = "auto",
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None: ...
    async def get_registry(self) -> dict[str, Any]: ...
    async def force_reindex(self, filename: str) -> None: ...
    async def upload_files(self, files: Any, prefix: str = "docs/") -> list[str]: ...
