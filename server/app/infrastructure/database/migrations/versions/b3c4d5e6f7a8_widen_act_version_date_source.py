"""Widen act_versions.date_source to VARCHAR(32).

The value 'extracted_trusted' (17 chars) overflows VARCHAR(16).

Revision ID: b3c4d5e6f7a8
Revises: a9b0c1d2e3f4
Create Date: 2026-09-08
"""

from alembic import op
import sqlalchemy as sa

revision = "b3c4d5e6f7a8"
down_revision = "a9b0c1d2e3f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "act_versions",
        "date_source",
        existing_type=sa.String(16),
        type_=sa.String(32),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "act_versions",
        "date_source",
        existing_type=sa.String(32),
        type_=sa.String(16),
        existing_nullable=False,
    )
