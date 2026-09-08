"""Add composite unique index for cursor pagination on chunks.

Replaces the 2-column UNIQUE (document_id, chunk_index) with a 3-column
UNIQUE (document_id, chunk_index, id) so that keyset pagination queries
like ``WHERE (chunk_index, id) > (?, ?) ORDER BY chunk_index, id`` are
covered by a single index scan.

Revision ID: a9b0c1d2e3f4
Revises: a3b4c5d6e7f8
Create Date: 2026-09-08
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a9b0c1d2e3f4"
down_revision: str | Sequence[str] | None = "a3b4c5d6e7f8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ux_chunks_document_index", "chunks", type_="unique")
    op.create_unique_constraint(
        "ux_chunks_document_chunk_index_id",
        "chunks",
        ["document_id", "chunk_index", "id"],
    )


def downgrade() -> None:
    op.drop_constraint("ux_chunks_document_chunk_index_id", "chunks", type_="unique")
    op.create_unique_constraint(
        "ux_chunks_document_index",
        "chunks",
        ["document_id", "chunk_index"],
    )
