"""Durable benchmark retry/import identity and artifact metadata."""

from alembic import op
import sqlalchemy as sa

revision = "r8h9i0j1k2l3"
down_revision = "q7h8i9j0k1l2"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "benchmark_checkpoints",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.add_column("benchmark_runs", sa.Column("job_id", sa.Integer(), nullable=True))
    op.add_column("benchmark_runs", sa.Column("legacy_hash", sa.String(64), nullable=True))
    op.add_column("benchmark_runs", sa.Column("artifacts", sa.JSON(), nullable=False, server_default="{}"))
    op.create_unique_constraint("uq_benchmark_runs_job_id", "benchmark_runs", ["job_id"])
    op.create_unique_constraint("uq_benchmark_runs_legacy_hash", "benchmark_runs", ["legacy_hash"])


def downgrade():
    op.drop_column("benchmark_checkpoints", "updated_at")
    op.drop_constraint("uq_benchmark_runs_legacy_hash", "benchmark_runs", type_="unique")
    op.drop_constraint("uq_benchmark_runs_job_id", "benchmark_runs", type_="unique")
    for column in ("artifacts", "legacy_hash", "job_id"):
        op.drop_column("benchmark_runs", column)
