"""Enforce unique chunk positions; existing duplicates abort without changing data."""

from alembic import op


revision = "t0j1k2l3m4n5"
down_revision = "s9i0j1k2l3m4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint("ux_chunks_document_index", "chunks", ["document_id", "chunk_index"])


def downgrade() -> None:
    op.drop_constraint("ux_chunks_document_index", "chunks", type_="unique")
