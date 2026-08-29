"""CLI command: Document indexing in Qdrant (S3-only)."""

from __future__ import annotations

import asyncio
import logging
import sys

import typer
from config import settings
from infrastructure.services.ingestion_service import IngestionService
from infrastructure.storage import get_storage

logger = logging.getLogger("cli")

ingest_app = typer.Typer(help="Document indexing in Qdrant (S3 storage)")


def _create_service(uow_factory=None) -> IngestionService:
    """Create IngestionService with proper dependencies (no DB in CLI mode)."""
    from infrastructure.repositories.qdrant_vector_store_repository import QdrantVectorStoreRepository
    from infrastructure.storage import LazyStorage

    return IngestionService(
        vector_store_repo=QdrantVectorStoreRepository(),
        file_storage=LazyStorage(),
        uow_factory=uow_factory,
    )


@ingest_app.command("run")
def ingest_run(
    docs_dir: str = typer.Option("docs/", "--docs-dir", "-d", help="S3 prefix (default: docs/)"),
    reset: bool = typer.Option(False, "--reset", help="Reset collection and registry, reindex everything"),
    domain: str = typer.Option("auto", "--domain", help="Document domain: auto, legal, general"),
) -> None:
    """Full indexing of documents from S3 bucket."""
    try:
        service = _create_service()
        asyncio.run(service.run_full_ingestion(docs_dir=docs_dir, reset=reset, domain=domain))
    except Exception as exc:
        logger.error("Indexing error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("file")
def ingest_file(
    file_path: str = typer.Argument(..., help="S3 key (e.g. docs/report.pdf)"),
    force: bool = typer.Option(False, "--force", help="Reindex even if file is already in registry"),
    domain: str = typer.Option("auto", "--domain", help="Document domain: auto, legal, general"),
) -> None:
    """Add a single file from S3 to existing collection."""
    try:
        service = _create_service()

        async def _run():
            if force:
                await service.force_reindex(file_path.split("/")[-1])
            await service.run_single_file(file_path, domain=domain)

        asyncio.run(_run())
    except Exception as exc:
        logger.error("File indexing error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("upload")
def ingest_upload(
    file_path: str = typer.Argument(..., help="Local file path to upload to S3"),
    key: str = typer.Option(None, "--key", "-k", help="S3 key (default: docs/<filename>)"),
) -> None:
    """Upload a local file to S3 storage."""
    try:
        from pathlib import Path as _Path

        storage = get_storage()

        path = _Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", file_path)
            sys.exit(1)

        s3_key = key or f"docs/{path.name}"
        data = path.read_bytes()
        asyncio.run(storage.upload_file(s3_key, data))
        logger.info("Uploaded: s3://%s/%s (%d bytes)", settings.s3_bucket, s3_key, len(data))
    except Exception as exc:
        logger.error("Upload error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("list")
def ingest_list() -> None:
    """Show list of indexed files."""
    try:
        service = _create_service()
        registry = asyncio.run(service._registry_list_all())

        if not registry:
            logger.info("Registry empty — no files indexed.")
            return

        logger.info("Indexed files: %d", len(registry))
        logger.info("%-50s  %6s  %8s  %s", "File", "Chunks", "Chars", "Date")
        logger.info("-" * 85)
        for name, meta in sorted(registry.items()):
            logger.info(
                "%-50s  %6s  %8s  %s",
                name[:50],
                meta.get("chunks", "?"),
                f"{meta.get('chars', 0):,}",
                meta.get("indexed_at", "?")[:19] if meta.get("indexed_at") else "?",
            )
    except Exception as exc:
        logger.error("Registry listing error", exc_info=exc)
        sys.exit(1)
