"""add chunks table with pg_trgm for exact substring search

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-08-07 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Enable pg_trgm extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # 2. Create chunks table
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("creation_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False, server_default=""),
        sa.Column("visibility", sa.String(20), nullable=False),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "group_id",
            sa.Integer(),
            sa.ForeignKey("groups.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # 3. Indexes
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_visibility", "chunks", ["visibility"])
    op.create_index("ix_chunks_owner_id", "chunks", ["owner_id"], postgresql_where="owner_id IS NOT NULL")
    op.create_index("ix_chunks_group_id", "chunks", ["group_id"], postgresql_where="group_id IS NOT NULL")

    # 4. GIN trigram index for substring search (CONCURRENTLY not possible inside a transaction,
    #    but this runs once at migration time with no live traffic expected)
    op.execute("CREATE INDEX ix_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops)")


def downgrade() -> None:
    op.drop_table("chunks")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
