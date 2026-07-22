"""Use Case: RunIngestion — full document folder ingestion."""

from __future__ import annotations

from domain.repositories.ingestion_registry_repository import IngestionRegistryRepository


class RunIngestion:
    def __init__(
        self,
        registry_repo: IngestionRegistryRepository,
        ingestion_service,
    ) -> None:
        self._registry_repo = registry_repo
        self._ingestion_service = ingestion_service

    def execute(self, docs_dir: str, reset: bool = False, prefix: str | None = None) -> None:
        self._ingestion_service.run_full_ingestion(docs_dir, reset=reset, prefix=prefix)
