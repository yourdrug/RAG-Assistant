"""Add vector_store_outbox table for saga/outbox pattern (Postgres ↔ Qdrant consistency).

Revision ID: t2u3v4w5x6y7
Revises: r7s8t9u0v1w2
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "t2u3v4w5x6y7"
down_revision: str | Sequence[str] | None = "r7s8t9u0v1w2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add 'indexing' status to documents check constraint
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_status_check")
    op.execute(
        "ALTER TABLE documents ADD CONSTRAINT documents_status_check "
        "CHECK (status IN ('pending', 'processing', 'indexing', 'done', 'failed'))"
    )

    op.create_table(
        "vector_store_outbox",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "creation_date",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("operation", sa.String(30), nullable=False),
        sa.Column("aggregate_type", sa.String(20), nullable=False, server_default="document"),
        sa.Column("aggregate_id", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="8"),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("locked_by", sa.String(64), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'in_progress', 'done', 'failed', 'dead_letter')",
            name="vector_store_outbox_status_check",
        ),
        sa.CheckConstraint(
            "operation IN ('upsert_chunks', 'delete_by_document', 'delete_chunks')",
            name="vector_store_outbox_operation_check",
        ),
    )
    op.create_index(
        "idx_outbox_dispatch",
        "vector_store_outbox",
        ["status", "next_attempt_at"],
        postgresql_where="status IN ('pending', 'failed')",
    )
    op.create_index(
        "idx_outbox_aggregate",
        "vector_store_outbox",
        ["aggregate_type", "aggregate_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_outbox_aggregate", table_name="vector_store_outbox")
    op.drop_index("idx_outbox_dispatch", table_name="vector_store_outbox")
    op.drop_table("vector_store_outbox")

    # Revert documents check constraint
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_status_check")
    op.execute(
        "ALTER TABLE documents ADD CONSTRAINT documents_status_check "
        "CHECK (status IN ('pending', 'processing', 'done', 'failed'))"
    )
