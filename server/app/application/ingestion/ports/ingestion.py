"""Public operations of the ingestion feature."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from domain.knowledge.enums import DocumentVisibility, DomainDetectionMode


if TYPE_CHECKING:
    from collections.abc import Sequence

    from application.ingestion.dto import IngestRegistryResult, IngestStatusResult
    from application.ingestion.ports.upload import UploadFile


@runtime_checkable
class IngestionPort(Protocol):
    def resolve_docs_dir(self, docs_dir: str) -> str: ...
    async def run_full(
        self,
        docs_dir: str | None = None,
        *,
        reset: bool = False,
        domain: str = DomainDetectionMode.AUTO.value,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> IngestStatusResult: ...
    def resolve_ingest_target(self, file_path: str) -> str: ...
    async def run_single(
        self,
        file_path: str,
        *,
        force: bool = False,
        domain: str = DomainDetectionMode.AUTO.value,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> IngestStatusResult: ...
    async def get_registry(self) -> IngestRegistryResult: ...
    async def force_reindex(self, filename: str) -> None: ...
    async def upload_files(self, files: Sequence[UploadFile], prefix: str = "docs/") -> list[str]: ...
