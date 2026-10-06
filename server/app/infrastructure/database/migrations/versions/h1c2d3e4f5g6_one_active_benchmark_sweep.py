"""Enforce one pending/running benchmark sweep across all processes.

Revision ID: h1c2d3e4f5g6
Revises: g0b1c2d3e4f5
"""

import sqlalchemy as sa
from alembic import op

revision = "h1c2d3e4f5g6"
down_revision = "g0b1c2d3e4f5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Fail if existing data violates the invariant: cancelling running jobs
    # here would silently change business behavior. Resolve them explicitly.
    op.create_index(
        "uq_benchmark_sweeps_one_active",
        "benchmark_sweeps",
        [sa.text("(1)")],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("uq_benchmark_sweeps_one_active", table_name="benchmark_sweeps")
