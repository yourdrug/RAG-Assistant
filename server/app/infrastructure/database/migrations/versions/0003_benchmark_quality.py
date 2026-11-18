"""Benchmark datasets are named `<pack>-<purpose>`; acceptance policies and human labels.

Renaming has no aliases: questions, sweeps, runs and the dataset names stored inside run
results are renamed together, so a regression check still finds its baseline. Restore an
old question dump before this revision, or downgrade to 0002, restore and upgrade again.
"""

import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


RENAMES = {
    "main": "belarus_law-main",
    "quick-30": "belarus_law-quick",
    "synthetic-support-v1": "support-synthetic",
    "synthetic-internal-v1": "internal-synthetic",
    "chunking-contract": "belarus_law-chunking-contract",
    "chunking-general": "belarus_law-chunking-general",
    "chunking-decree": "belarus_law-chunking-decree",
    "chunking-legal": "belarus_law-chunking-legal",
}
DATASET_TABLES = ("benchmark_questions", "benchmark_sweeps", "benchmark_runs")


def rename_datasets(renames: dict[str, str], default: str) -> None:
    connection = op.get_bind()

    for table in DATASET_TABLES:
        op.alter_column(table, "dataset", server_default=sa.text(f"'{default}'::character varying"))

        for old, new in renames.items():
            connection.execute(
                # Table names come from the constant above, values are bound parameters.
                sa.text(f"UPDATE {table} SET dataset = :new WHERE dataset = :old"),  # noqa: S608
                {"old": old, "new": new},
            )

    runs = connection.execute(sa.text("SELECT id, summary_metrics, per_question_results FROM benchmark_runs"))

    for run_id, summary, results in runs.all():
        changed = False

        for row in results or []:
            if isinstance(row, dict) and row.get("dataset") in renames:
                row["dataset"] = renames[row["dataset"]]
                changed = True

        by_dataset = (summary or {}).get("by_dataset")

        if isinstance(by_dataset, dict) and set(by_dataset) & set(renames):
            summary["by_dataset"] = {renames.get(key, key): value for key, value in by_dataset.items()}
            changed = True

        if changed:
            connection.execute(
                sa.text(
                    "UPDATE benchmark_runs SET summary_metrics = CAST(:summary AS json), "
                    "per_question_results = CAST(:results AS json) WHERE id = :id"
                ),
                {"id": run_id, "summary": json.dumps(summary), "results": json.dumps(results)},
            )


def upgrade() -> None:
    rename_datasets(RENAMES, "belarus_law-main")
    op.create_table(
        "benchmark_dataset_policies",
        sa.Column("dataset", sa.VARCHAR(length=100), autoincrement=False, nullable=False),
        sa.Column(
            "thresholds",
            postgresql.JSONB,
            server_default=sa.text("'{}'::jsonb"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("noise", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.Column("calibration", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.Column("updated_by", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "update_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            name="benchmark_dataset_policies_updated_by_fkey",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("dataset", name="benchmark_dataset_policies_pkey"),
    )
    op.create_table(
        "benchmark_human_labels",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("question_key", sa.VARCHAR(length=200), autoincrement=False, nullable=False),
        sa.Column("metric", sa.VARCHAR(length=20), autoincrement=False, nullable=False),
        sa.Column("score", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
        sa.Column("labeler_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("note", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column(
            "update_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.CheckConstraint(
            "metric::text = ANY (ARRAY['correctness'::character varying, "
            "'faithfulness'::character varying]::text[])",
            name="ck_benchmark_human_labels_metric",
        ),
        sa.CheckConstraint(
            "score >= 0::double precision AND score <= 10::double precision",
            name="ck_benchmark_human_labels_score",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["benchmark_runs.id"], name="benchmark_human_labels_run_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["labeler_id"], ["users.id"], name="benchmark_human_labels_labeler_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="benchmark_human_labels_pkey"),
        sa.UniqueConstraint(
            "run_id", "question_key", "metric", "labeler_id", name="uq_benchmark_human_labels_case"
        ),
    )


def downgrade() -> None:
    op.drop_table("benchmark_human_labels")
    op.drop_table("benchmark_dataset_policies")
    rename_datasets({new: old for old, new in RENAMES.items()}, "main")
