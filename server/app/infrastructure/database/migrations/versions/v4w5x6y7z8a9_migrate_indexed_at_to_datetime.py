"""migrate indexed_at to DateTime

Revision ID: v4w5x6y7z8a9
Revises: u3v4w5x6y7z8
Create Date: 2026-09-03
"""

from alembic import op
import sqlalchemy as sa


revision = "v4w5x6y7z8a9"
down_revision = "u3v4w5x6y7z8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "ingestion_registry",
        "indexed_at",
        type_=sa.DateTime(),
        postgresql_using="indexed_at::timestamp",
    )


def downgrade() -> None:
    op.alter_column(
        "ingestion_registry",
        "indexed_at",
        type_=sa.String(50),
        postgresql_using="indexed_at::text",
    )
