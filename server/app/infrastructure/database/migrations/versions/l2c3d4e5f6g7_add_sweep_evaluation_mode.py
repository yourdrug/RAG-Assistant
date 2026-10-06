"""Persist fast/full evaluation scope; existing sweeps remain fast."""

from alembic import op
import sqlalchemy as sa

from domain.value_objects.sweep_evaluation_mode import SweepEvaluationMode
from domain.value_objects.benchmark_strategy import BenchmarkStrategy

revision = "l2c3d4e5f6g7"
down_revision = "j1c2d3e4f5g6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "benchmark_sweeps",
        sa.Column(
            "evaluation_mode", sa.String(20), nullable=False, server_default=SweepEvaluationMode.FAST.value
        ),
    )
    op.create_check_constraint(
        "benchmark_sweeps_evaluation_mode_check",
        "benchmark_sweeps",
        f"evaluation_mode IN ('{SweepEvaluationMode.FAST.value}', '{SweepEvaluationMode.FULL.value}')",
    )
    op.create_check_constraint(
        "benchmark_sweeps_full_grid_check",
        "benchmark_sweeps",
        f"evaluation_mode != '{SweepEvaluationMode.FULL.value}' OR strategy = '{BenchmarkStrategy.GRID.value}'",
    )


def downgrade() -> None:
    op.drop_constraint("benchmark_sweeps_full_grid_check", "benchmark_sweeps", type_="check")
    op.drop_constraint("benchmark_sweeps_evaluation_mode_check", "benchmark_sweeps", type_="check")
    op.drop_column("benchmark_sweeps", "evaluation_mode")
