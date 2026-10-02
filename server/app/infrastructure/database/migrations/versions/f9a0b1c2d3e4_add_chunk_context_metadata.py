"""Persist chunk interpretation context for PostgreSQL retrieval.

Revision ID: f9a0b1c2d3e4
Revises: e8f9a0b1c2d3
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "f9a0b1c2d3e4"
down_revision = "e8f9a0b1c2d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("context_metadata", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_check_constraint(
        "chk_chunks_context_metadata_object", "chunks", "jsonb_typeof(context_metadata) = 'object'"
    )


def downgrade() -> None:
    op.drop_constraint("chk_chunks_context_metadata_object", "chunks", type_="check")
    op.drop_column("chunks", "context_metadata")
