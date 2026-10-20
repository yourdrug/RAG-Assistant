"""CLI command: Document indexing in Qdrant (S3-only)."""

from __future__ import annotations

from domain.value_objects.doc_domain import DomainDetectionMode

import asyncio
import logging
import sys
from pathlib import Path

from application.budget import billing_principal
from composition.infrastructure import InfrastructureContainer
from composition.service_providers import create_ingestion_service
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client

import typer
from config import settings
from domain.value_objects.visibility import DocumentVisibility
from application.ingestion import IngestionService

logger = logging.getLogger("cli")

ingest_app = typer.Typer(help="Document indexing in Qdrant (S3 storage)")


def create_service(infra) -> IngestionService:
    """Create IngestionService with proper dependencies via composition."""
    return create_ingestion_service(infra)


async def run_with_infrastructure(action):
    """Start and release the Redis/DB resources required by composition."""
    infra = InfrastructureContainer()
    redis_started = False
    database_started = False
    try:
        await redis_client.init()
        redis_started = True
        await database.connect()
        database_started = True
        infra.init(database)
        billing_principal.set("system:ingest-cli")
        await infra.bootstrap_configuration()
        return await action(infra)
    finally:
        try:
            await infra.dispose()
        finally:
            try:
                if database_started:
                    await database.disconnect()
            finally:
                if redis_started:
                    await redis_client.aclose()


@ingest_app.command("run")
def ingest_run(
    docs_dir: str = typer.Option("docs/", "--docs-dir", "-d", help="S3 prefix (default: docs/)"),
    reset: bool = typer.Option(
        False, "--reset", help="Reindex selected prefix; preserve existing document IDs and assignments"
    ),
    domain: str = typer.Option(
        DomainDetectionMode.AUTO.value, "--domain", help="Document domain: auto, legal, general"
    ),
    visibility: str = typer.Option(
        DocumentVisibility.INTERNAL_PUBLIC.value,
        "--visibility",
        "-v",
        help=(
            "Visibility for new documents; existing assignments are preserved: "
            f"{', '.join(v.value for v in DocumentVisibility)}"
        ),
    ),
    group_id: int = typer.Option(None, "--group-id", help="Group ID (required if visibility=internal_group)"),
    client_id: int = typer.Option(
        None, "--client-id", help="Client ID (required if visibility=client_private)"
    ),
) -> None:
    """Full indexing of documents from S3 bucket."""
    vis = DocumentVisibility.validate(visibility)
    try:

        async def run_action(infra) -> None:
            await create_service(infra).run_full(
                docs_dir=docs_dir,
                reset=reset,
                domain=domain,
                visibility=vis,
                group_id=group_id,
                client_id=client_id,
            )

        asyncio.run(run_with_infrastructure(run_action))
    except Exception as exc:
        logger.error("Indexing error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("file")
def ingest_file(
    file_path: str = typer.Argument(..., help="S3 key (e.g. docs/report.pdf)"),
    force: bool = typer.Option(False, "--force", help="Reindex even if file is already in registry"),
    domain: str = typer.Option(
        DomainDetectionMode.AUTO.value, "--domain", help="Document domain: auto, legal, general"
    ),
    visibility: str = typer.Option(
        DocumentVisibility.INTERNAL_PUBLIC.value,
        "--visibility",
        "-v",
        help=(
            "Visibility for new documents; existing assignments are preserved: "
            f"{', '.join(v.value for v in DocumentVisibility)}"
        ),
    ),
    group_id: int = typer.Option(None, "--group-id", help="Group ID (required if visibility=internal_group)"),
    client_id: int = typer.Option(
        None, "--client-id", help="Client ID (required if visibility=client_private)"
    ),
) -> None:
    """Add a single file from S3 to existing collection."""
    vis = DocumentVisibility.validate(visibility)
    try:

        async def run_action(infra):
            service = create_service(infra)
            await service.run_single(
                file_path, force=force, domain=domain, visibility=vis, group_id=group_id, client_id=client_id
            )

        asyncio.run(run_with_infrastructure(run_action))
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
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", file_path)
            sys.exit(1)

        s3_key = key or f"docs/{path.name}"
        data = path.read_bytes()

        async def upload_action(infra):
            await infra.ml.storage.upload_file(s3_key, data)

        asyncio.run(run_with_infrastructure(upload_action))
        logger.info("Uploaded: s3://%s/%s (%d bytes)", settings.s3_bucket, s3_key, len(data))
    except Exception as exc:
        logger.error("Upload error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("list")
def ingest_list() -> None:
    """Show list of indexed files."""
    try:

        async def run_action(infra):
            return await create_service(infra).get_registry()

        registry = asyncio.run(run_with_infrastructure(run_action))

        if not registry.files:
            logger.info("Registry empty — no files indexed.")
            return

        logger.info("Indexed files: %d", registry.total_files)
        logger.info("%-50s  %6s  %8s  %s", "File", "Chunks", "Chars", "Date")
        logger.info("-" * 85)
        for item in registry.files:
            logger.info(
                "%-50s  %6s  %8s  %s",
                item.ingestion_key,
                item.chunks,
                f"{item.chars:,}",
                item.indexed_at.isoformat()[:19] if item.indexed_at else "?",
            )
    except Exception as exc:
        logger.error("Registry listing error", exc_info=exc)
        sys.exit(1)
