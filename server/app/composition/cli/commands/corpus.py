"""CLI command: offline maintenance of the active Qdrant corpus."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Annotated

from qdrant_client import QdrantClient
import typer

from config import MLProviderSection, VectorStoreSection, process_section
from domain.shared.exceptions import ValidationError
from infrastructure.corpus.clients import raw_corpus_client
from infrastructure.corpus.control import CORPUS_HEARTBEAT_TTL_SECONDS
from infrastructure.corpus.repository import CorpusPhase, CorpusRepository
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.corpus.sparse_migration import (
    SparseMigrationReport,
    has_current_sparse,
    migrate_sparse_vectors,
)
from infrastructure.corpus.validation import validate_corpus
from infrastructure.database.database import database
from infrastructure.ml.config.embedding_identity import configured_embedding_identity
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME


logger = logging.getLogger("cli")

corpus_app = typer.Typer(help="Offline maintenance of the active Qdrant corpus")


def configured_snapshot(client: QdrantClient) -> CorpusSnapshot:
    """Snapshot for a deployment whose corpus journal is not initialized yet."""
    settings = process_section(VectorStoreSection)
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    return CorpusSnapshot(
        settings.qdrant_url,
        settings.collection_name,
        aliases.get(settings.collection_name, settings.collection_name),
        configured_embedding_identity(),
    )


async def active_snapshot(repository: CorpusRepository) -> CorpusSnapshot | None:
    """Journal snapshot after checking that no API/worker process can write to the corpus."""
    state = await repository.read_state()

    if state is None:
        return None

    if state["phase"] != CorpusPhase.ACTIVE:
        message = f"Corpus journal is in phase {state['phase']}; finish or fail the switch first"
        raise ValidationError(message)

    alive = [p["role"] for p in await repository.participants() if p["alive"]]

    if alive:
        # Processes stopped just before this command keep a fresh heartbeat for one TTL.
        logger.info("Waiting for heartbeats of %s to expire ...", sorted(alive))
        await asyncio.sleep(CORPUS_HEARTBEAT_TTL_SECONDS + 1)
        alive = [p["role"] for p in await repository.participants() if p["alive"]]

    if alive:
        message = f"Stop API and worker before migrating the corpus (alive: {sorted(alive)})"
        raise ValidationError(message)

    return CorpusSnapshot(**state["active"])


async def migrate_sparse(*, dry_run: bool) -> SparseMigrationReport | None:
    repository = CorpusRepository(database.get_write_session)
    journal = await active_snapshot(repository)
    endpoint = journal.endpoint if journal else process_section(VectorStoreSection).qdrant_url
    client = raw_corpus_client(endpoint)
    try:
        snapshot = journal if journal is not None else await asyncio.to_thread(configured_snapshot, client)
        logger.info("Active corpus: %s (alias %s)", snapshot.physical_collection, snapshot.alias)

        if dry_run:
            info = await asyncio.to_thread(client.get_collection, snapshot.physical_collection)
            logger.info(
                "points=%s sparse=%s current=%s (target %s)",
                info.points_count,
                sorted(info.config.params.sparse_vectors or {}),
                has_current_sparse(info),
                SPARSE_VECTOR_NAME,
            )
            return None

        report = await asyncio.to_thread(migrate_sparse_vectors, client, snapshot)
        # Same point-level check the API runs at startup: every PostgreSQL chunk is present
        # with matching ACL metadata and a BM25 vector.
        rows = await repository.chunks()
        await asyncio.to_thread(
            validate_corpus,
            client,
            snapshot.endpoint,
            snapshot.alias,
            snapshot.embedding_identity,
            process_section(MLProviderSection).embed_dim,
            rows=rows,
        )
        logger.info("Verified %d PostgreSQL chunks against %s", len(rows), snapshot.physical_collection)
        return report
    finally:
        await asyncio.to_thread(client.close)


async def run_with_database(*, dry_run: bool) -> SparseMigrationReport | None:
    await database.connect()
    try:
        return await migrate_sparse(dry_run=dry_run)
    finally:
        await database.disconnect()


@corpus_app.command("migrate-sparse")
def corpus_migrate_sparse(
    *,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the current sparse index only")] = False,
) -> None:
    """Re-encode BM25 sparse vectors in place after a SPARSE_VECTOR_NAME change.

    Run with API and worker stopped, before starting the new version. Dense vectors,
    ids and payload are kept; the command is restartable and ends with a point-level
    check against PostgreSQL.
    """
    try:
        report = asyncio.run(run_with_database(dry_run=dry_run))
    except Exception as exc:
        logger.exception("Sparse migration failed", exc_info=exc)
        sys.exit(1)

    if report is not None:
        logger.info(
            "%s: %d points, %s",
            report.collection,
            report.points,
            "rebuilt" if report.rebuilt else "already current",
        )
