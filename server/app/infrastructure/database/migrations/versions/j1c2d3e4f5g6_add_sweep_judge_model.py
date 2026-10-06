"""Persist the judge model selected for a benchmark sweep."""

from alembic import op
import sqlalchemy as sa

revision = "j1c2d3e4f5g6"
down_revision = "h1c2d3e4f5g6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_sweeps", sa.Column("judge_model", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("benchmark_sweeps", "judge_model")
