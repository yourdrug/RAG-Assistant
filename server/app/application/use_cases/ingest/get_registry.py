"""Use Case: GetIngestRegistry — view indexed files."""

from __future__ import annotations

from domain.repositories.ingestion_registry_repository import IngestionRegistryRepository

from application.dto.ingest_dto import IngestRegistryItemDTO, IngestRegistryResult


class GetIngestRegistry:
    def __init__(self, registry_repo: IngestionRegistryRepository) -> None:
        self._registry_repo = registry_repo

    def execute(self) -> IngestRegistryResult:
        registry = self._registry_repo.load()
        items = [
            IngestRegistryItemDTO(
                filename=name,
                chunks=meta.get("chunks", 0),
                chars=meta.get("chars", 0),
                indexed_at=meta.get("indexed_at", ""),
                source=meta.get("source", ""),
            )
            for name, meta in sorted(registry.items())
        ]
        return IngestRegistryResult(
            total_files=len(items),
            total_chunks=sum(i.chunks for i in items),
            files=items,
        )
