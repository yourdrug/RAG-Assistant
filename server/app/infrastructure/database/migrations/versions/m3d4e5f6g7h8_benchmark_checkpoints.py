"""Store resumable benchmark stages in PostgreSQL."""

from alembic import op
import sqlalchemy as sa

revision = "m3d4e5f6g7h8"
down_revision = "l2c3d4e5f6g7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_checkpoints",
        sa.Column(
            "sweep_id", sa.Integer(), sa.ForeignKey("benchmark_sweeps.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("key", sa.String(512), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("sweep_id", "key"),
    )


def downgrade() -> None:
    op.drop_table("benchmark_checkpoints")
