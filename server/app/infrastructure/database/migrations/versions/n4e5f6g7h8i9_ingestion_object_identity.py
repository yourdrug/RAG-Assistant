"""Use exact object keys for ingestion registry and document identity."""

import logging

from alembic import op
import sqlalchemy as sa

from domain.value_objects.source_type import SourceType
from domain.entities.vector_outbox_entry import OutboxOperation

revision = "n4e5f6g7h8i9"
down_revision = "m3d4e5f6g7h8"
branch_labels = None
depends_on = None

log = logging.getLogger("default")


def upgrade() -> None:
    connection = op.get_bind()
    invalid = connection.execute(
        sa.text("SELECT filename FROM ingestion_registry WHERE source !~ '^s3://[^/]+/.+$' LIMIT 1")
    ).scalar_one_or_none()
    if invalid is not None:
        raise RuntimeError(f"Invalid legacy S3 source for ingestion registry entry {invalid!r}")

    op.add_column("ingestion_registry", sa.Column("object_key", sa.Text(), nullable=True))
    connection.execute(
        sa.text("UPDATE ingestion_registry SET object_key = regexp_replace(source, '^s3://[^/]+/', '')")
    )
    duplicate = connection.execute(
        sa.text("SELECT object_key FROM ingestion_registry GROUP BY object_key HAVING count(*) > 1 LIMIT 1")
    ).scalar_one_or_none()
    if duplicate is not None:
        raise RuntimeError(
            f"Duplicate legacy ingestion object key {duplicate!r}; resolve bucket conflicts first"
        )

    # CLI ingestion historically left source_path empty. API uploads have a
    # storage path and cannot be adopted merely because their name matches.
    ambiguous = connection.execute(
        sa.text(
            "SELECT r.object_key FROM ingestion_registry r JOIN documents d ON "
            "d.filename = r.filename AND d.source_path = '' AND d.source_type = :file_type "
            "GROUP BY r.object_key HAVING count(*) > 1 LIMIT 1"
        ),
        {"file_type": SourceType.FILE.value},
    ).scalar_one_or_none()
    if ambiguous is not None:
        raise RuntimeError(
            f"Legacy ingestion document binding is ambiguous for {ambiguous!r}; resolve it before migrating"
        )

    candidates = (
        connection.execute(
            sa.text(
                "SELECT d.id, latest.source, latest.source ~ '^s3://[^/]+/.+$' AS confirmed, "
                "regexp_replace(latest.source, '^s3://[^/]+/', '') AS object_key "
                "FROM documents d LEFT JOIN LATERAL ("
                " SELECT payload->>'source' AS source FROM vector_store_outbox o "
                " WHERE o.aggregate_type = 'document' AND o.aggregate_id = d.id AND o.operation = :operation "
                " ORDER BY o.id DESC LIMIT 1"
                ") latest ON true "
                "WHERE d.source_path = '' AND d.source_type = :file_type "
                "AND EXISTS (SELECT 1 FROM ingestion_registry r WHERE r.filename = d.filename)"
            ),
            {"file_type": SourceType.FILE.value, "operation": OutboxOperation.SET_DOCUMENT_ID.value},
        )
        .mappings()
        .all()
    )
    bindings = []
    for candidate in candidates:
        if not candidate["confirmed"]:
            raise RuntimeError(
                f"Missing committed S3 source evidence for legacy ingestion document {candidate['id']}"
            )
        bindings.append({"document_id": candidate["id"], "object_key": candidate["object_key"]})

    op.add_column("documents", sa.Column("ingestion_key", sa.Text(), nullable=True))
    if bindings:
        connection.execute(
            sa.text("UPDATE documents SET ingestion_key = :object_key WHERE id = :document_id"), bindings
        )
    invalidated = connection.execute(
        sa.text(
            "UPDATE ingestion_registry r SET file_hash = '', indexed_at = NULL "
            "WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.ingestion_key = r.object_key)"
        )
    ).rowcount
    if invalidated:
        log.warning("Invalidated %d legacy ingestion registry entries without a document", invalidated)

    op.alter_column("ingestion_registry", "object_key", nullable=False)
    op.alter_column("ingestion_registry", "source", type_=sa.Text(), existing_type=sa.String(1000))
    op.drop_index("idx_ingestion_registry_filename", table_name="ingestion_registry")
    op.create_index("ux_ingestion_registry_object_key", "ingestion_registry", ["object_key"], unique=True)
    op.create_index(
        "ux_documents_ingestion_key",
        "documents",
        [sa.text("COALESCE(owner_id, 0)"), sa.text("COALESCE(group_id, 0)"), "ingestion_key"],
        unique=True,
        postgresql_where=sa.text("ingestion_key IS NOT NULL"),
    )


def downgrade() -> None:
    duplicate = (
        op.get_bind()
        .execute(
            sa.text("SELECT filename FROM ingestion_registry GROUP BY filename HAVING count(*) > 1 LIMIT 1")
        )
        .scalar_one_or_none()
    )
    if duplicate is not None:
        raise RuntimeError(
            f"Cannot downgrade ingestion registry with duplicate basenames: {duplicate!r}; resolve them first"
        )
    long_source = (
        op.get_bind()
        .execute(sa.text("SELECT object_key FROM ingestion_registry WHERE length(source) > 1000 LIMIT 1"))
        .scalar_one_or_none()
    )
    if long_source is not None:
        raise RuntimeError(f"Cannot downgrade long ingestion source {long_source!r} to the legacy schema")
    op.drop_index("ux_documents_ingestion_key", table_name="documents")
    op.drop_column("documents", "ingestion_key")
    op.drop_index("ux_ingestion_registry_object_key", table_name="ingestion_registry")
    op.create_index("idx_ingestion_registry_filename", "ingestion_registry", ["filename"], unique=True)
    op.drop_column("ingestion_registry", "object_key")
    op.alter_column("ingestion_registry", "source", type_=sa.String(1000), existing_type=sa.Text())
