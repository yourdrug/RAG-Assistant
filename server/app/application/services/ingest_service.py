"""Application Service: IngestService — orchestrates ingestion use cases."""

from __future__ import annotations

from application.dto.ingest_dto import IngestRegistryResult, IngestStatusResult
from application.use_cases.ingest.get_registry import GetIngestRegistry
from application.use_cases.ingest.ingest_single_file import IngestSingleFile
from application.use_cases.ingest.run_ingestion import RunIngestion


class IngestAppService:
    def __init__(
        self,
        run_ingestion: RunIngestion,
        ingest_single_file: IngestSingleFile,
        get_registry: GetIngestRegistry,
        path_resolver,
    ) -> None:
        self._run_ingestion = run_ingestion
        self._ingest_single_file = ingest_single_file
        self._get_registry = get_registry
        self._path_resolver = path_resolver

    def run_full(self, docs_dir: str, reset: bool = False) -> IngestStatusResult:
        resolved_dir = self._path_resolver.resolve_docs_dir(docs_dir)
        self._run_ingestion.execute(resolved_dir, reset=reset)
        mode = "RESET + full reindex" if reset else "APPEND (new files only)"
        return IngestStatusResult(status="started", mode=mode, docs_dir=resolved_dir)

    def run_single(self, file_path: str, force: bool = False) -> IngestStatusResult:
        resolved = self._path_resolver.resolve_ingest_target(file_path)
        self._ingest_single_file.execute(resolved, force=force)
        return IngestStatusResult(status="started", file=resolved, force=force)

    def get_registry(self) -> IngestRegistryResult:
        return self._get_registry.execute()

    def resolve_docs_dir(self, docs_dir: str) -> str:
        return self._path_resolver.resolve_docs_dir(docs_dir)

    def resolve_ingest_target(self, file_path: str) -> str:
        return self._path_resolver.resolve_ingest_target(file_path)

    def force_reindex(self, filename: str) -> None:
        self._path_resolver.force_reindex(filename)
