"""Application service for document ingestion orchestration.

Delegates the heavy lifting (parsing, embedding, vector upload) to the
``IngestionPort`` while managing DB-side concerns: status queries, registry
lookups, and force-reindex operations.  Each method that touches the
database opens its own UnitOfWork.
"""

from __future__ import annotations

from application.dto.ingest_dto import IngestRegistryItemDTO, IngestRegistryResult, IngestStatusResult
from application.ports.ingestion_port import IngestionPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory


class IngestAppService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        ingestion_service: IngestionPort,
    ) -> None:
        self._uow_factory = uow_factory
        self._ingestion = ingestion_service

    async def run_full(
        self, docs_dir: str | None = None, reset: bool = False, domain: str = "auto"
    ) -> IngestStatusResult:
        resolved = self._ingestion.resolve_docs_dir(docs_dir or "docs/")
        await self._ingestion.run_full_ingestion(resolved, reset=reset, domain=domain)
        mode = "RESET + full reindex" if reset else "APPEND (new files only)"
        return IngestStatusResult(status="started", mode=mode, docs_dir=resolved)

    async def run_single(
        self, file_path: str, force: bool = False, domain: str = "auto"
    ) -> IngestStatusResult:
        resolved = self._ingestion.resolve_ingest_target(file_path)
        await self._ingestion.run_single_file(resolved, domain=domain)
        return IngestStatusResult(status="started", file=resolved, force=force)

    async def get_registry(self) -> IngestRegistryResult:
        raw = await self._ingestion.get_registry()
        items = []
        for filename, info in raw.items():
            items.append(
                IngestRegistryItemDTO(
                    filename=filename,
                    chunks=info.get("chunks", 0),
                    chars=info.get("chars", 0),
                    indexed_at=info.get("indexed_at", ""),
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
        return self._ingestion.resolve_docs_dir(prefix)

    def resolve_ingest_target(self, file_path: str) -> str:
        return self._ingestion.resolve_ingest_target(file_path)

    async def force_reindex(self, filename: str) -> None:
        await self._ingestion.force_reindex(filename)

    async def upload_files(self, files, prefix: str = "docs/") -> list[str]:
        return await self._ingestion.upload_files(files, prefix)
