"""Cancellation, per-file checkpoints and an index for the durable relay."""

from alembic import op
import sqlalchemy as sa


revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A concurrent index commits preceding DDL. A failed build must be retryable
    # even when Alembic has not yet recorded this revision.
    op.execute("ALTER TABLE background_jobs ADD COLUMN IF NOT EXISTS progress JSONB")
    op.execute("ALTER TABLE background_jobs DROP CONSTRAINT IF EXISTS background_jobs_status_check")
    op.create_check_constraint(
        "background_jobs_status_check",
        "background_jobs",
        "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
    )
    with op.get_context().autocommit_block():
        valid = op.get_bind().scalar(
            sa.text(
                "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
                "JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND c.relname='idx_background_jobs_relay'"
            )
        )

        if valid is False:
            op.drop_index(
                "idx_background_jobs_relay", table_name="background_jobs", postgresql_concurrently=True
            )

        op.create_index(
            "idx_background_jobs_relay",
            "background_jobs",
            ["heartbeat_at", "id"],
            postgresql_where=sa.text("status = 'pending' AND dispatch_payload IS NOT NULL"),
            postgresql_concurrently=True,
            if_not_exists=True,
        )


def downgrade() -> None:
    # Explicit downgrade preserves cancellation as a terminal state of the old release.
    op.execute("UPDATE background_jobs SET status = 'failed' WHERE status = 'cancelled'")
    with op.get_context().autocommit_block():
        op.drop_index("idx_background_jobs_relay", table_name="background_jobs", postgresql_concurrently=True)
    op.drop_constraint("background_jobs_status_check", "background_jobs", type_="check")
    op.create_check_constraint(
        "background_jobs_status_check",
        "background_jobs",
        "status IN ('pending', 'running', 'done', 'failed')",
    )
    op.drop_column("background_jobs", "progress")
