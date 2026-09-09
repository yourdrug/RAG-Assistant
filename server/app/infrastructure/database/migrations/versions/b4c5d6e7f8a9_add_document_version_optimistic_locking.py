"""add document version for optimistic locking

Revision ID: b4c5d6e7f8a9
Revises: x1a2b3c4d5e6
Create Date: 2026-09-09
"""

from alembic import op
import sqlalchemy as sa

revision = "b4c5d6e7f8a9"
down_revision = "x1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("version", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("documents", "version")
