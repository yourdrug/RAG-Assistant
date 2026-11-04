"""CLI command: Document indexing in Qdrant (S3-only)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging
from pathlib import Path
import sys
from typing import Annotated

import typer

from application.budget import billing_principal
from application.ingestion import IngestionService
from application.ingestion.dto import IngestRegistryResult
from composition.infrastructure import InfrastructureContainer
from composition.service_providers import create_ingestion_service
from config import settings
from domain.knowledge.enums import DocumentVisibility, DomainDetectionMode
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client


logger = logging.getLogger("cli")

ingest_app = typer.Typer(help="Document indexing in Qdrant (S3 storage)")


def create_service(infra: InfrastructureContainer) -> IngestionService:
    """Create IngestionService with proper dependencies via composition."""
    return create_ingestion_service(infra)


async def run_with_infrastructure[T](action: Callable[[InfrastructureContainer], Awaitable[T]]) -> T:
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
    *,
    docs_dir: Annotated[str, typer.Option("--docs-dir", "-d", help="S3 prefix (default: docs/)")] = "docs/",
    reset: Annotated[
        bool,
        typer.Option(
            "--reset", help="Reindex selected prefix; preserve existing document IDs and assignments"
        ),
    ] = False,
    domain: Annotated[
        str, typer.Option("--domain", help="Document domain: auto, legal, general")
    ] = DomainDetectionMode.AUTO.value,
    visibility: Annotated[
        str,
        typer.Option(
            "--visibility",
            "-v",
            help=(
                "Visibility for new documents; existing assignments are preserved: "
                f"{', '.join(v.value for v in DocumentVisibility)}"
            ),
        ),
    ] = DocumentVisibility.INTERNAL_PUBLIC.value,
    group_id: Annotated[
        int | None, typer.Option("--group-id", help="Group ID (required if visibility=internal_group)")
    ] = None,
    client_id: Annotated[
        int | None, typer.Option("--client-id", help="Client ID (required if visibility=client_private)")
    ] = None,
) -> None:
    """Full indexing of documents from S3 bucket."""
    vis = DocumentVisibility.parse(visibility)
    try:

        async def run_action(infra: InfrastructureContainer) -> None:
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
        logger.exception("Indexing error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("file")
def ingest_file(
    file_path: Annotated[str, typer.Argument(help="S3 key (e.g. docs/report.pdf)")],
    *,
    force: Annotated[
        bool, typer.Option("--force", help="Reindex even if file is already in registry")
    ] = False,
    domain: Annotated[
        str, typer.Option("--domain", help="Document domain: auto, legal, general")
    ] = DomainDetectionMode.AUTO.value,
    visibility: Annotated[
        str,
        typer.Option(
            "--visibility",
            "-v",
            help=(
                "Visibility for new documents; existing assignments are preserved: "
                f"{', '.join(v.value for v in DocumentVisibility)}"
            ),
        ),
    ] = DocumentVisibility.INTERNAL_PUBLIC.value,
    group_id: Annotated[
        int | None, typer.Option("--group-id", help="Group ID (required if visibility=internal_group)")
    ] = None,
    client_id: Annotated[
        int | None, typer.Option("--client-id", help="Client ID (required if visibility=client_private)")
    ] = None,
) -> None:
    """Add a single file from S3 to existing collection."""
    vis = DocumentVisibility.parse(visibility)
    try:

        async def run_action(infra: InfrastructureContainer) -> None:
            service = create_service(infra)
            await service.run_single(
                file_path, force=force, domain=domain, visibility=vis, group_id=group_id, client_id=client_id
            )

        asyncio.run(run_with_infrastructure(run_action))
    except Exception as exc:
        logger.exception("File indexing error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("upload")
def ingest_upload(
    file_path: Annotated[str, typer.Argument(help="Local file path to upload to S3")],
    *,
    key: Annotated[str | None, typer.Option("--key", "-k", help="S3 key (default: docs/<filename>)")] = None,
) -> None:
    """Upload a local file to S3 storage."""
    try:
        path = Path(file_path)

        if not path.exists():
            logger.error("File not found: %s", file_path)
            sys.exit(1)

        s3_key = key or f"docs/{path.name}"
        data = path.read_bytes()

        async def upload_action(infra: InfrastructureContainer) -> None:
            await infra.ml.storage.upload_file(s3_key, data)

        asyncio.run(run_with_infrastructure(upload_action))
        logger.info("Uploaded: s3://%s/%s (%d bytes)", settings.s3_bucket, s3_key, len(data))
    except Exception as exc:
        logger.exception("Upload error", exc_info=exc)
        sys.exit(1)


@ingest_app.command("list")
def ingest_list() -> None:
    """Show list of indexed files."""
    try:

        async def run_action(infra: InfrastructureContainer) -> IngestRegistryResult:
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
        logger.exception("Registry listing error", exc_info=exc)
        sys.exit(1)
