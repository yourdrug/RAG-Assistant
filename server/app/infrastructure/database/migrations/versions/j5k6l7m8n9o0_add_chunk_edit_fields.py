"""add chunk edit fields and document source_type

Revision ID: j5k6l7m8n9o0
Revises: i4j5k6l7m8n9
Create Date: 2026-08-14 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "j5k6l7m8n9o0"
down_revision: str | Sequence[str] | None = "i4j5k6l7m8n9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ChunkModel: add edited_at, edited_by, manual
    op.add_column(
        "chunks",
        sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "edited_by",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "chunks",
        sa.Column("manual", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index("ix_chunks_edited_by", "chunks", ["edited_by"], postgresql_where="edited_by IS NOT NULL")

    # DocumentModel: add source_type, has_manual_edits
    op.add_column(
        "documents",
        sa.Column("source_type", sa.String(16), nullable=False, server_default="file"),
    )
    op.add_column(
        "documents",
        sa.Column("has_manual_edits", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_check_constraint(
        "documents_source_type_check",
        "documents",
        "source_type IN ('file', 'manual')",
    )
    op.create_index("idx_documents_source_type", "documents", ["source_type"])


def downgrade() -> None:
    op.drop_index("idx_documents_source_type", table_name="documents")
    op.drop_constraint("documents_source_type_check", "documents")
    op.drop_column("documents", "has_manual_edits")
    op.drop_column("documents", "source_type")

    op.drop_index("ix_chunks_edited_by", table_name="chunks")
    op.drop_column("chunks", "manual")
    op.drop_column("chunks", "edited_by")
    op.drop_column("chunks", "edited_at")
