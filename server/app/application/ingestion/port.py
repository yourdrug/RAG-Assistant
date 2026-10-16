"""IngestionPort -- application-layer abstraction for the S3-only document ingestion pipeline.

Defines the protocol that ``IngestAppService`` depends on.  The concrete
implementation lives in ``application.ingestion.orchestrator`` and
uses S3 as the sole file storage backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from domain.value_objects.visibility import DocumentVisibility


if TYPE_CHECKING:
    from collections.abc import Sequence

    from application.ingestion.dto import RegistryRecord, UploadFile


@runtime_checkable
class IngestionPort(Protocol):
    def resolve_docs_dir(self, docs_dir: str) -> str: ...
    async def run_full_ingestion(
        self,
        docs_dir: str | None = None,
        *,
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
    async def get_registry(self) -> dict[str, RegistryRecord]: ...
    async def force_reindex(self, filename: str) -> None: ...
    async def upload_files(self, files: Sequence[UploadFile], prefix: str = "docs/") -> list[str]: ...
