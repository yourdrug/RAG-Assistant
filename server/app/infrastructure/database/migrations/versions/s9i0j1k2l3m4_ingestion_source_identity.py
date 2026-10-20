"""Identify ingestion sources by full object key; run with ingestion stopped.

Only confirmed legacy links are transferred. Ambiguity aborts the transaction.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from alembic import op
import sqlalchemy as sa


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy import RowMapping


revision = "s9i0j1k2l3m4"
down_revision = "r8h9i0j1k2l3"
branch_labels = None
depends_on = None

STATUSES = "status IN ('pending', 'processing', 'indexing', 'done', 'failed')"


def source_key(source: str) -> str:
    if not source.startswith("s3://"):
        message = f"Cannot identify legacy ingestion source: {source!r}"
        raise ValueError(message)

    bucket, separator, key = source[5:].partition("/")

    if not bucket or not separator or not key or key.startswith("/") or ".." in PurePosixPath(key).parts:
        message = f"Invalid legacy ingestion source: {source!r}"
        raise ValueError(message)

    return key


def document_candidates(
    documents: Sequence[RowMapping], entries: list[tuple[RowMapping, str]]
) -> dict[int, set[str]]:
    by_filename: dict[str, set[str]] = defaultdict(set)

    for row, key in entries:
        by_filename[row["filename"]].add(key)

    candidates: dict[int, set[str]] = defaultdict(set)

    for doc in documents:
        if doc["source_type"] == "manual":
            continue

        if doc["source_path"].startswith("s3://"):
            candidates[doc["id"]].add(source_key(doc["source_path"]))
        elif not doc["source_path"]:
            candidates[doc["id"]].update(by_filename[doc["filename"]])

    return candidates


def add_outbox_candidates(
    connection: sa.Connection, candidates: dict[int, set[str]], documents: Sequence[RowMapping]
) -> None:
    events = (
        connection.execute(
            sa.text(
                "SELECT aggregate_id, operation, payload FROM vector_store_outbox WHERE aggregate_type = 'document'"
            )
        )
        .mappings()
        .all()
    )
    by_id = {doc["id"]: doc for doc in documents}

    for event in events:
        target = by_id.get(event["aggregate_id"])

        if target is None or target["source_type"] == "manual":
            continue

        payload = event["payload"]
        sources = [payload.get("source", "")]
        sources.extend(point.get("metadata", {}).get("source", "") for point in payload.get("points", []))

        for source in sources:
            if source.startswith("s3://"):
                candidates[target["id"]].add(source_key(source))


def validate_candidates(candidates: dict[int, set[str]], documents: Sequence[RowMapping]) -> None:
    by_id = {doc["id"]: doc for doc in documents}
    slots = set()

    for doc_id, choices in candidates.items():
        if len(choices) > 1:
            message = f"Ambiguous legacy ingestion document {doc_id}: {sorted(choices)}"
            raise ValueError(message)

        if choices:
            doc = by_id[doc_id]
            slot = (next(iter(choices)), doc["owner_id"], doc["group_id"])

            if slot in slots:
                message = f"Duplicate legacy ingestion slot: {slot}"
                raise ValueError(message)

            slots.add(slot)


def upgrade() -> None:
    connection = op.get_bind()
    op.execute("LOCK TABLE documents, ingestion_registry, vector_store_outbox IN ACCESS EXCLUSIVE MODE")
    registry = (
        connection.execute(sa.text("SELECT id, filename, source FROM ingestion_registry")).mappings().all()
    )
    documents = (
        connection.execute(
            sa.text(
                "SELECT id, filename, source_path, source_type, owner_id, group_id, status FROM documents"
            )
        )
        .mappings()
        .all()
    )
    entries = [(row, source_key(row["source"])) for row in registry]
    keys = [key for _, key in entries]

    if len(keys) != len(set(keys)):
        message = "Ambiguous legacy registry: duplicate ingestion keys"
        raise ValueError(message)

    candidates = document_candidates(documents, entries)
    add_outbox_candidates(connection, candidates, documents)
    validate_candidates(candidates, documents)

    op.add_column("documents", sa.Column("ingestion_key", sa.Text(), nullable=True))
    op.add_column("ingestion_registry", sa.Column("ingestion_key", sa.Text(), nullable=True))

    for row, key in entries:
        connection.execute(
            sa.text("UPDATE ingestion_registry SET ingestion_key = :key WHERE id = :id"),
            {"key": key, "id": row["id"]},
        )

    for doc_id, choices in candidates.items():
        if choices:
            connection.execute(
                sa.text("UPDATE documents SET ingestion_key = :key WHERE id = :id"),
                {"key": next(iter(choices)), "id": doc_id},
            )

    op.alter_column("ingestion_registry", "ingestion_key", nullable=False)
    op.drop_index("idx_ingestion_registry_filename", table_name="ingestion_registry")
    op.create_index(
        "idx_ingestion_registry_key",
        "ingestion_registry",
        ["ingestion_key"],
        unique=True,
    )
    op.drop_index("ux_documents_active_slot", table_name="documents")
    op.execute(
        "CREATE UNIQUE INDEX ux_documents_active_slot ON documents (COALESCE(owner_id, 0), COALESCE(group_id, 0), filename) WHERE ingestion_key IS NULL AND "
        + STATUSES
    )
    op.execute(
        "CREATE UNIQUE INDEX ux_documents_ingestion_slot ON documents (ingestion_key, COALESCE(owner_id, 0), COALESCE(group_id, 0)) WHERE ingestion_key IS NOT NULL AND "
        + STATUSES
    )


def downgrade() -> None:
    # Duplicate basenames must be resolved before returning to the old schema.
    op.execute("LOCK TABLE documents, ingestion_registry IN ACCESS EXCLUSIVE MODE")
    op.drop_index("ux_documents_active_slot", table_name="documents")
    op.execute(
        "CREATE UNIQUE INDEX ux_documents_active_slot ON documents (COALESCE(owner_id, 0), COALESCE(group_id, 0), filename) WHERE "
        + STATUSES
    )
    op.create_index(
        "idx_ingestion_registry_filename",
        "ingestion_registry",
        ["filename"],
        unique=True,
    )
    op.drop_index("ux_documents_ingestion_slot", table_name="documents")
    op.drop_index("idx_ingestion_registry_key", table_name="ingestion_registry")
    op.drop_column("documents", "ingestion_key")
    op.drop_column("ingestion_registry", "ingestion_key")
