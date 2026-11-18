"""Every saved judge calibration is kept; the dataset policy holds only the latest one.

Upgrade copies the calibrations already stored in policies into the history.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_judge_calibrations",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("dataset", sa.VARCHAR(length=100), autoincrement=False, nullable=False),
        sa.Column("run_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("judge_model", sa.VARCHAR(length=200), autoincrement=False, nullable=True),
        sa.Column("judge_rubric_version", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("metrics", postgresql.JSONB, autoincrement=False, nullable=False),
        sa.Column("created_by", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "measured_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        # History outlives the run it was measured on.
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["benchmark_runs.id"],
            name="benchmark_judge_calibrations_run_id_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name="benchmark_judge_calibrations_created_by_fkey",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="benchmark_judge_calibrations_pkey"),
    )
    op.create_index(
        "idx_benchmark_judge_calibrations_dataset",
        "benchmark_judge_calibrations",
        ["dataset", sa.text("measured_at DESC")],
    )
    op.execute(
        """
        INSERT INTO benchmark_judge_calibrations
            (dataset, run_id, judge_model, judge_rubric_version, metrics, created_by, measured_at)
        SELECT p.dataset,
               r.id,
               p.calibration ->> 'judge_model',
               p.calibration ->> 'judge_rubric_version',
               COALESCE(p.calibration -> 'metrics', '{}'::jsonb),
               u.id,
               (p.calibration ->> 'measured_at')::timestamptz
        FROM benchmark_dataset_policies p
        LEFT JOIN benchmark_runs r ON r.id = (p.calibration ->> 'run_id')::int
        LEFT JOIN users u ON u.id = p.updated_by
        WHERE p.calibration IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_index("idx_benchmark_judge_calibrations_dataset", table_name="benchmark_judge_calibrations")
    op.drop_table("benchmark_judge_calibrations")
