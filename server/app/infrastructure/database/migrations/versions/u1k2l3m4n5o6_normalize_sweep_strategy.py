"""Normalize the former TPE alias before removing it from the application."""

from alembic import op


revision = "u1k2l3m4n5o6"
down_revision = "t0j1k2l3m4n5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE benchmark_sweeps SET strategy = 'optuna_tpe' WHERE strategy = 'successive_halving'")


def downgrade() -> None:
    # Both values executed TPE; the original spelling cannot be recovered.
    pass
