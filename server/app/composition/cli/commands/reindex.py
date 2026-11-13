"""CLI command: re-chunk stored documents with the current pipeline.

Documents keep their ids, ACL and stored domain; old chunks are replaced through
the vector outbox (``DELETE_CHUNKS`` + ``UPSERT_CHUNKS``), which the worker relays
to Qdrant. Progress is written to a checkpoint file after every document, so an
interrupted run resumes from the last processed id.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Annotated

import typer

from application.documents.reindex import ReindexOutcome, ReindexRequest
from composition.cli.commands.ingest import run_with_infrastructure
from composition.service_providers import create_reindex_service
from domain.knowledge.enums import DomainKey


if TYPE_CHECKING:
    from application.documents.reindex import ReindexEvent, ReindexReport
    from composition.infrastructure import InfrastructureContainer


logger = logging.getLogger("cli")

reindex_app = typer.Typer(help="Re-chunk stored documents with the current chunking code")


def read_checkpoint(path: Path) -> tuple[int, list[int]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0, []

    last_id = data.get("last_id")
    failed = data.get("failed", [])

    if not isinstance(last_id, int) or last_id < 0 or not all(isinstance(item, int) for item in failed):
        message = f"Invalid checkpoint {path}: expected integer last_id and failed ids"
        raise typer.BadParameter(message)

    return last_id, failed


class CheckpointWriter:
    """Persist the last processed id atomically after every document."""

    def __init__(self, path: Path | None, scope: dict[str, object], failed: list[int]) -> None:
        self.path = path
        self.scope = scope
        self.failed = failed

    def __call__(self, event: ReindexEvent) -> None:
        logger.info(
            "doc %d %-15s %s %s", event.document_id, event.outcome, event.filename, event.detail or ""
        )

        if event.outcome == ReindexOutcome.FAILED:
            self.failed.append(event.document_id)

        if self.path is None:
            return

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({**self.scope, "last_id": event.document_id, "failed": self.failed}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def log_summary(report: ReindexReport) -> None:
    for outcome in (
        ReindexOutcome.SELECTED,
        ReindexOutcome.REINDEXED,
        ReindexOutcome.FAILED,
        ReindexOutcome.SKIPPED_MANUAL,
        ReindexOutcome.SKIPPED_STATUS,
        ReindexOutcome.SKIPPED_SOURCE,
    ):
        if ids := report.ids(outcome):
            logger.info("%s: %d %s", outcome, len(ids), ids)

    if manual := report.ids(ReindexOutcome.SKIPPED_MANUAL):
        logger.warning("Documents with manual chunk edits were not re-chunked: %s", manual)

    logger.info("last_id: %s", report.last_id)


@reindex_app.command("run")
def reindex_run(  # noqa: PLR0913 - each Typer option is an independent CLI argument.
    *,
    domain: Annotated[
        str | None, typer.Option("--domain", help="Only documents with this stored doc_domain")
    ] = None,
    document_ids: Annotated[
        list[int] | None, typer.Option("--id", help="Only these document ids (repeatable)")
    ] = None,
    after_id: Annotated[
        int | None, typer.Option("--after-id", help="Start after this id (overrides the checkpoint)")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Process at most N documents")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Documents per batch")] = 10,
    pause: Annotated[
        float, typer.Option("--pause", min=0.0, help="Seconds between batches (lets the worker drain)")
    ] = 0.0,
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", help="JSON progress file; resumes from its last_id")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="List selected documents only")] = False,
) -> None:
    """Re-chunk documents in id order with batches, pauses and resumable progress."""
    doc_domain = DomainKey(domain) if domain else None
    saved_id, failed = read_checkpoint(checkpoint) if checkpoint else (0, [])
    start = after_id if after_id is not None else saved_id
    request = ReindexRequest(
        doc_domain=doc_domain,
        document_ids=tuple(document_ids or ()),
        after_id=start,
        limit=limit,
        batch_size=batch_size,
        pause_seconds=pause,
        dry_run=dry_run,
    )
    writer = CheckpointWriter(None if dry_run else checkpoint, {"doc_domain": doc_domain}, failed)
    logger.info("Reindex from id > %d (domain=%s, dry_run=%s)", start, doc_domain or "all", dry_run)

    async def run_action(infra: InfrastructureContainer) -> ReindexReport:
        return await create_reindex_service(infra).run(request, writer)

    try:
        report = asyncio.run(run_with_infrastructure(run_action))
    except Exception as exc:
        logger.exception("Reindex error", exc_info=exc)
        sys.exit(1)

    log_summary(report)

    if report.ids(ReindexOutcome.FAILED):
        sys.exit(1)
