"""add benchmark_questions, benchmark_sweeps, benchmark_runs tables

Revision ID: k6l7m8n9o0p1
Revises: 69930d1c204f
Create Date: 2026-08-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "k6l7m8n9o0p1"
down_revision: str | Sequence[str] | None = "69930d1c204f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "benchmark_questions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "creation_date",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("source_hint", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("dataset", sa.String(100), nullable=False, server_default="main"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_by",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
    )

    op.create_index("idx_benchmark_questions_dataset", "benchmark_questions", ["dataset"])
    op.create_index("idx_benchmark_questions_is_active", "benchmark_questions", ["is_active"])

    op.create_table(
        "benchmark_sweeps",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "creation_date",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("search_space", sa.JSON(), nullable=False),
        sa.Column("objective_weights", sa.JSON(), nullable=False),
        sa.Column("dataset", sa.String(100), nullable=False, server_default="main"),
        sa.Column("top_n_llm", sa.Integer(), nullable=False, server_default="3"),
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("background_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("total_configs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("evaluated_configs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("best_run_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
            name="benchmark_sweeps_status_check",
        ),
    )

    op.create_index("idx_benchmark_sweeps_status", "benchmark_sweeps", ["status"])

    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "creation_date",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "sweep_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_sweeps.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("summary_metrics", sa.JSON(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=False, server_default="0"),
        sa.Column("llm_evaluated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("dataset", sa.String(100), nullable=False, server_default="main"),
        sa.Column("per_question_results", sa.JSON(), nullable=True),
        sa.Column("filename", sa.String(255), nullable=True),
    )

    op.create_index("idx_benchmark_runs_sweep_id", "benchmark_runs", ["sweep_id"])
    op.create_index("idx_benchmark_runs_dataset", "benchmark_runs", ["dataset"])


def downgrade() -> None:
    op.drop_index("idx_benchmark_runs_dataset", table_name="benchmark_runs")
    op.drop_index("idx_benchmark_runs_sweep_id", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
    op.drop_index("idx_benchmark_sweeps_status", table_name="benchmark_sweeps")
    op.drop_table("benchmark_sweeps")
    op.drop_index("idx_benchmark_questions_is_active", table_name="benchmark_questions")
    op.drop_index("idx_benchmark_questions_dataset", table_name="benchmark_questions")
    op.drop_table("benchmark_questions")
