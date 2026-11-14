"""Version durable payloads and fence execution attempts."""

from alembic import op
import sqlalchemy as sa


revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for column in (
        sa.Column("payload_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("deduplication_key", sa.String(128)),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column("attempt_id", sa.String(36)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("dispatch_attempts", sa.Integer, nullable=False, server_default="0"),
    ):
        op.add_column("background_jobs", column)

    # Build without blocking writes of the running release (expand/contract deploy).
    with op.get_context().autocommit_block():
        op.create_index(
            "ux_background_jobs_active_dedup",
            "background_jobs",
            ["deduplication_key"],
            unique=True,
            postgresql_where=sa.text("deduplication_key IS NOT NULL AND status IN ('pending', 'running')"),
            postgresql_concurrently=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ux_background_jobs_active_dedup",
            table_name="background_jobs",
            postgresql_concurrently=True,
            if_exists=True,
        )

    for name in (
        "dispatch_attempts",
        "next_attempt_at",
        "lease_until",
        "attempt_id",
        "attempt",
        "deduplication_key",
        "payload_version",
    ):
        op.drop_column("background_jobs", name)
