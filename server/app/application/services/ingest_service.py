"""Application Service: IngestAppService — manages ingestion via UoWFactory.

Each method that touches DB opens its own UnitOfWork.
No db/session parameters.
"""

from __future__ import annotations

from infrastructure.services.ingestion_service import IngestionService
from infrastructure.uow_factory import UnitOfWorkFactory

from application.dto.ingest_dto import IngestRegistryItemDTO, IngestRegistryResult, IngestStatusResult


class IngestAppService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        ingestion_service: IngestionService,
    ) -> None:
        self._uow_factory = uow_factory
        self._ingestion = ingestion_service

    def run_full(self, docs_dir: str, reset: bool = False) -> IngestStatusResult:
        resolved_dir = self._ingestion.resolve_docs_dir(docs_dir)
        self._ingestion.run_full_ingestion(resolved_dir, reset=reset)
        mode = "RESET + full reindex" if reset else "APPEND (new files only)"
        return IngestStatusResult(status="started", mode=mode, docs_dir=resolved_dir)

    def run_single(self, file_path: str, force: bool = False) -> IngestStatusResult:
        resolved = self._ingestion.resolve_ingest_target(file_path)
        self._ingestion.run_single_file(resolved)
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

    def upload_files(self, files, prefix: str = "docs/") -> list[str]:
        return self._ingestion.upload_files(files, prefix)
