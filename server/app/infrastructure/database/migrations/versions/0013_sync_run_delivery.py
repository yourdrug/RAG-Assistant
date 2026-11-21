"""Webhook delivery of a sync run: a redelivered webhook never starts a second run."""

from alembic import op
import sqlalchemy as sa


revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("sync_runs", sa.Column("delivery_id", sa.String(length=64), nullable=True))
    op.create_index(
        "ux_sync_runs_delivery",
        "sync_runs",
        ["channel_id", "delivery_id"],
        unique=True,
        postgresql_where="delivery_id IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_index("ux_sync_runs_delivery", table_name="sync_runs")
    op.drop_column("sync_runs", "delivery_id")
