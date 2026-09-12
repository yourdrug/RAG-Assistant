"""add version column to benchmark_sweeps for optimistic locking

Revision ID: a1b2c3d4e5f8
Revises: d7e8f9a0b1c2
Create Date: 2026-09-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f8"
down_revision: Union[str, Sequence[str], None] = "d7e8f9a0b1c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "benchmark_sweeps",
        sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("benchmark_sweeps", "version")
