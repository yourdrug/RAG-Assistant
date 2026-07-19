"""
CLI command: Document indexing in Qdrant.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

logger = logging.getLogger("cli")

ingest_app = typer.Typer(help="Document indexing in Qdrant")


@ingest_app.command("run")
def ingest_run(
    docs_dir: str = typer.Option(
        str(Path("/code/project/data") / "docs_sample"),
        "--docs-dir",
        "-d",
        help="Document folder (local mode)",
    ),
    reset: bool = typer.Option(False, "--reset", help="Reset collection and registry, reindex everything"),
    s3: bool = typer.Option(False, "--s3", help="Index from S3 (instead of local folder)"),
    prefix: str = typer.Option("docs/", "--prefix", "-p", help="S3 prefix (default: docs/)"),
) -> None:
    """Full indexing of document folder."""
    try:
        from config import settings
        from services.ingest_service import IngestService

        if s3:
            settings.file_backend = "s3"

        service = IngestService()
        service.run_full_ingestion(docs_dir, reset=reset, prefix=prefix)
    except Exception as exc:
        logger.error("Indexing error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("file")
def ingest_file(
    file_path: str = typer.Argument(..., help="File path (local) or S3 key (docs/report.pdf)"),
    force: bool = typer.Option(False, "--force", help="Reindex even if file is already in registry"),
    s3: bool = typer.Option(False, "--s3", help="Treat path as S3 key"),
) -> None:
    """Add a single file to existing collection."""
    try:
        from config import settings
        from services.ingest_service import IngestService

        if s3:
            settings.file_backend = "s3"

        service = IngestService()

        if force:
            service.force_reindex(Path(file_path).name)

        service.run_single_file(file_path)
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
        from config import settings
        from infrastructure.storage import get_storage

        settings.file_backend = "s3"
        storage = get_storage()

        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", file_path)
            sys.exit(1)

        s3_key = key or f"docs/{path.name}"
        data = path.read_bytes()
        storage.upload_file(s3_key, data)
        logger.info("Uploaded: s3://%s/%s (%d bytes)", settings.s3_bucket, s3_key, len(data))
    except Exception as exc:
        logger.error("Upload error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("list")
def ingest_list() -> None:
    """Show list of indexed files."""
    try:
        from services.ingest_service import IngestService

        service = IngestService()
        registry = service.get_registry()

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
                meta.get("indexed_at", "?")[:19],
            )
    except Exception as exc:
        logger.error("Registry error", exc_info=exc)
        sys.exit(1)
