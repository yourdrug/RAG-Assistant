"""Use Case: IngestSingleFile — index a single file."""

from __future__ import annotations

from pathlib import Path

from domain.repositories.ingestion_registry_repository import IngestionRegistryRepository


class IngestSingleFile:
    def __init__(
        self,
        registry_repo: IngestionRegistryRepository,
        ingestion_service,
    ) -> None:
        self._registry_repo = registry_repo
        self._ingestion_service = ingestion_service

    def execute(self, file_path: str, force: bool = False) -> None:
        if force:
            self._ingestion_service.force_reindex(Path(file_path).name)
        self._ingestion_service.run_single_file(file_path)
