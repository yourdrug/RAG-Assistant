"""Add content_hash column to chunks table

Revision ID: p5q6r7s8t9u0
Revises: o4p5q6r7s8t9
Create Date: 2026-08-29
"""

from alembic import op
import sqlalchemy as sa


revision = "p5q6r7s8t9u0"
down_revision = "o4p5q6r7s8t9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("content_hash", sa.String(16), nullable=True))
    op.create_index("idx_chunks_content_hash", "chunks", ["content_hash"])


def downgrade() -> None:
    op.drop_index("idx_chunks_content_hash", table_name="chunks")
    op.drop_column("chunks", "content_hash")
