"""add failed to active slot index

Revision ID: u3v4w5x6y7z8
Revises: t2u3v4w5x6y7
Create Date: 2026-09-03
"""

from alembic import op
import sqlalchemy as sa


revision = "u3v4w5x6y7z8"
down_revision = "t2u3v4w5x6y7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index(
        "ux_documents_active_slot",
        table_name="documents",
        postgresql_where=sa.text("status IN ('pending', 'processing', 'done')"),
    )
    op.create_index(
        "ux_documents_active_slot",
        "documents",
        ["owner_id", "filename"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing', 'done', 'failed')"),
    )


def downgrade() -> None:
    op.drop_index(
        "ux_documents_active_slot",
        table_name="documents",
        postgresql_where=sa.text("status IN ('pending', 'processing', 'done', 'failed')"),
    )
    op.create_index(
        "ux_documents_active_slot",
        "documents",
        ["owner_id", "filename"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing', 'done')"),
    )
