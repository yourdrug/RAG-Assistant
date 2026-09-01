"""add config_parameters table for dynamic config without restart

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-30 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop table if it exists from a failed previous run
    op.execute("DROP TABLE IF EXISTS config_parameters CASCADE")

    op.create_table(
        "config_parameters",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("value_type", sa.String(length=20), server_default="str", nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("min_value", sa.Float(), nullable=True),
        sa.Column("max_value", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )

    op.create_index("idx_config_parameters_category", "config_parameters", ["category"])

    # Default values are seeded at startup from settings (initialization.py)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS config_parameters")
