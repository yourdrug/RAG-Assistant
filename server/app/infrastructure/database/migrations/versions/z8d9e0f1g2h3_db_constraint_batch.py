"""Database constraint batch (AUDIT.md medium-term item 14).

- M-2: config_parameters (key, domain_key) UNIQUE NULLS NOT DISTINCT
  (dedupe first — duplicate rows break every get_by_key query forever)
- M-11: documents active-slot unique index via COALESCE so public/group
  (NULL owner) documents are covered too
- L-11: chunks UNIQUE (document_id, chunk_index) — duplicate chunk_index
  from concurrent manual appends broke ordering/citations
- L-20: benchmark_sweeps.best_run_id FK (dangling ids cleaned first)
- L-15: naive timestamp columns → timestamptz (writers already use aware
  datetimes; naive/aware mix had inconsistent tz semantics)
- outbox operation CHECK now allows update_metadata / set_document_id
  (previously enqueueing those ops violated the constraint and rolled back
  the whole use case)

Revision ID: z8d9e0f1g2h3
Revises: y6b7c8d9e0f1
Create Date: 2026-09-04
"""

from collections.abc import Sequence

from alembic import op

revision: str = "z8d9e0f1g2h3"
down_revision: str | Sequence[str] | None = "y6b7c8d9e0f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- M-2: config_parameters unique with NULLS NOT DISTINCT ---------------
    # Dedupe first (keep lowest id): duplicates make get_by_key raise
    # MultipleResultsFound → every config read/update 500s.
    op.execute(
        """
        DELETE FROM config_parameters a
        USING config_parameters b
        WHERE a.key = b.key
          AND a.domain_key IS NOT DISTINCT FROM b.domain_key
          AND a.id > b.id
        """
    )
    op.execute("ALTER TABLE config_parameters DROP CONSTRAINT IF EXISTS ux_config_parameters_key_domain")
    op.execute(
        "ALTER TABLE config_parameters "
        "ADD CONSTRAINT ux_config_parameters_key_domain UNIQUE NULLS NOT DISTINCT (key, domain_key)"
    )

    # --- M-11: documents active slot covers NULL-owner rows too --------------
    # (owner_id, filename) with NULL owner never conflicts (NULLs distinct) —
    # two concurrent uploads of the same public filename both succeeded.
    op.execute("DROP INDEX IF EXISTS ux_documents_active_slot")
    op.execute(
        "CREATE UNIQUE INDEX ux_documents_active_slot ON documents "
        "(COALESCE(owner_id, 0), COALESCE(group_id, 0), filename) "
        "WHERE status IN ('pending', 'processing', 'done', 'failed')"
    )

    # --- L-11: chunks UNIQUE (document_id, chunk_index) ----------------------
    op.execute(
        """
        DELETE FROM chunks a
        USING chunks b
        WHERE a.document_id = b.document_id
          AND a.chunk_index = b.chunk_index
          AND a.id > b.id
        """
    )
    op.execute("ALTER TABLE chunks ADD CONSTRAINT ux_chunks_document_index UNIQUE (document_id, chunk_index)")

    # --- L-20: best_run_id referential integrity ------------------------------
    op.execute(
        """
        UPDATE benchmark_sweeps s
        SET best_run_id = NULL
        WHERE s.best_run_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM benchmark_runs r WHERE r.id = s.best_run_id)
        """
    )
    op.execute(
        "ALTER TABLE benchmark_sweeps "
        "ADD CONSTRAINT fk_benchmark_sweeps_best_run "
        "FOREIGN KEY (best_run_id) REFERENCES benchmark_runs(id) ON DELETE SET NULL"
    )

    # --- L-15: naive → timestamptz (writers use datetime.now(tz=UTC)) --------
    for table, column in [
        ("documents", "indexed_at"),
        ("api_keys", "revoked_at"),
        ("api_keys", "last_used_at"),
        ("background_jobs", "started_at"),
        ("background_jobs", "finished_at"),
        ("ingestion_registry", "indexed_at"),
    ]:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} TYPE timestamptz USING {column} AT TIME ZONE 'UTC'"
        )

    # --- outbox operation CHECK: allow all outbox ops -------------------------
    # 'update_metadata' (temporal acts) and 'set_document_id' (CLI ingest) were
    # rejected by the constraint → enqueue rolled back the whole transaction.
    op.execute(
        "ALTER TABLE vector_store_outbox DROP CONSTRAINT IF EXISTS vector_store_outbox_operation_check"
    )
    op.execute(
        "ALTER TABLE vector_store_outbox "
        "ADD CONSTRAINT vector_store_outbox_operation_check CHECK (operation IN ("
        "'upsert_chunks', 'delete_by_document', 'delete_chunks', 'update_metadata', 'set_document_id'))"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE vector_store_outbox DROP CONSTRAINT IF EXISTS vector_store_outbox_operation_check"
    )
    op.execute(
        "ALTER TABLE vector_store_outbox "
        "ADD CONSTRAINT vector_store_outbox_operation_check CHECK (operation IN ("
        "'upsert_chunks', 'delete_by_document', 'delete_chunks'))"
    )

    for table, column in [
        ("ingestion_registry", "indexed_at"),
        ("background_jobs", "finished_at"),
        ("background_jobs", "started_at"),
        ("api_keys", "last_used_at"),
        ("api_keys", "revoked_at"),
        ("documents", "indexed_at"),
    ]:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} TYPE timestamp USING {column} AT TIME ZONE 'UTC'"
        )

    op.execute("ALTER TABLE benchmark_sweeps DROP CONSTRAINT IF EXISTS fk_benchmark_sweeps_best_run")
    op.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS ux_chunks_document_index")
    op.execute("DROP INDEX IF EXISTS ux_documents_active_slot")
    op.execute(
        "CREATE UNIQUE INDEX ux_documents_active_slot ON documents (owner_id, filename) "
        "WHERE status IN ('pending', 'processing', 'done', 'failed')"
    )
    op.execute("ALTER TABLE config_parameters DROP CONSTRAINT IF EXISTS ux_config_parameters_key_domain")
    op.execute(
        "ALTER TABLE config_parameters ADD CONSTRAINT ux_config_parameters_key_domain UNIQUE (key, domain_key)"
    )
