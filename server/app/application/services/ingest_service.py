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

    async def run_full(self, docs_dir: str, reset: bool = False, domain: str = "auto") -> IngestStatusResult:
        resolved_dir = self._ingestion.resolve_docs_dir(docs_dir)
        await self._ingestion.run_full_ingestion(resolved_dir, reset=reset, domain=domain)
        mode = "RESET + full reindex" if reset else "APPEND (new files only)"
        return IngestStatusResult(status="started", mode=mode, docs_dir=resolved_dir)

    async def run_single(
        self, file_path: str, force: bool = False, domain: str = "auto"
    ) -> IngestStatusResult:
        resolved = self._ingestion.resolve_ingest_target(file_path)
        await self._ingestion.run_single_file(resolved, domain=domain)
        return IngestStatusResult(status="started", file=resolved, force=force)

    def get_registry(self) -> IngestRegistryResult:
        raw = self._ingestion.get_registry()
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

    def resolve_docs_dir(self, docs_dir: str) -> str:
        return self._ingestion.resolve_docs_dir(docs_dir)

    def resolve_ingest_target(self, file_path: str) -> str:
        return self._ingestion.resolve_ingest_target(file_path)

    def force_reindex(self, filename: str) -> None:
        self._ingestion.force_reindex(filename)

    async def upload_files(self, files, prefix: str = "docs/") -> list[str]:
        return await self._ingestion.upload_files(files, prefix)
