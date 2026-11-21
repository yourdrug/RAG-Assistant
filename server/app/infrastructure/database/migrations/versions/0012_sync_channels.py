"""Sync channels and runs; documents bound to a channel file take no filename slot."""

from datetime import datetime

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

SLOT_STATUSES = "status IN ('pending', 'processing', 'indexing', 'done', 'failed')"
SLOT_COLUMNS: list[str | sa.ColumnElement[object]] = [
    "space_id",
    sa.literal_column("COALESCE(owner_id, 0)"),
    sa.literal_column("COALESCE(group_id, 0)"),
    "filename",
]
VISIBILITIES = "'internal_public', 'internal_group', 'internal_private', 'client_private', 'client_public'"


def created_at() -> sa.Column[datetime]:
    return sa.Column(
        "creation_date", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "sync_channels",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        created_at(),
        sa.Column("space_id", sa.Integer(), nullable=False),
        sa.Column("storage_kind", sa.String(length=16), nullable=False),
        sa.Column("location", postgresql.JSONB, nullable=False),
        sa.Column("folder", sa.Text(), server_default="", nullable=False),
        sa.Column("exclusions", postgresql.JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("visibility", sa.String(length=20), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=True),
        sa.Column("group_id", sa.Integer(), nullable=True),
        sa.Column("responsible_user_id", sa.Integer(), nullable=True),
        sa.Column("generation", sa.Integer(), server_default="1", nullable=False),
        sa.CheckConstraint("storage_kind IN ('github')", name="sync_channels_storage_kind_check"),
        sa.CheckConstraint(f"visibility IN ({VISIBILITIES})", name="sync_channels_visibility_check"),
        sa.CheckConstraint("jsonb_typeof(location) = 'object'", name="sync_channels_location_object"),
        sa.CheckConstraint("jsonb_typeof(exclusions) = 'array'", name="sync_channels_exclusions_array"),
        sa.CheckConstraint("generation > 0", name="sync_channels_generation_positive"),
        sa.ForeignKeyConstraint(
            ["space_id"], ["knowledge_spaces.id"], name="sync_channels_space_id_fkey", ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="sync_channels_owner_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["group_id"], ["groups.id"], name="sync_channels_group_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["responsible_user_id"],
            ["users.id"],
            name="sync_channels_responsible_user_id_fkey",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="sync_channels_pkey"),
    )
    op.create_index("idx_sync_channels_space", "sync_channels", ["space_id"])
    op.create_table(
        "sync_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        created_at(),
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("trigger", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("storage_revision", sa.String(length=64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("outcomes", postgresql.JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'partial', 'failed', 'cancelled', 'rejected')",
            name="sync_runs_status_check",
        ),
        sa.CheckConstraint("trigger IN ('manual', 'webhook', 'schedule')", name="sync_runs_trigger_check"),
        sa.CheckConstraint("jsonb_typeof(outcomes) = 'array'", name="sync_runs_outcomes_array"),
        sa.ForeignKeyConstraint(
            ["channel_id"], ["sync_channels.id"], name="sync_runs_channel_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="sync_runs_pkey"),
    )
    op.create_index("idx_sync_runs_channel", "sync_runs", ["channel_id", "id"])

    op.add_column("documents", sa.Column("sync_channel_id", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("sync_path", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("storage_version", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("storage_revision", sa.String(length=64), nullable=True))
    op.create_foreign_key(
        "documents_sync_channel_id_fkey", "documents", "sync_channels", ["sync_channel_id"], ["id"]
    )
    op.create_check_constraint(
        "documents_sync_binding_check", "documents", "(sync_channel_id IS NULL) = (sync_path IS NULL)"
    )
    op.create_index(
        "ux_documents_sync_file",
        "documents",
        ["sync_channel_id", "sync_path"],
        unique=True,
        postgresql_where="sync_channel_id IS NOT NULL",
    )
    replace_active_slot(f"ingestion_key IS NULL AND sync_channel_id IS NULL AND {SLOT_STATUSES}")


def replace_active_slot(where: str) -> None:
    op.drop_index("ux_documents_active_slot", table_name="documents")
    op.create_index(
        "ux_documents_active_slot", "documents", SLOT_COLUMNS, unique=True, postgresql_where=where
    )


def downgrade() -> None:
    bound = op.get_bind().scalar(sa.text("SELECT count(*) FROM documents WHERE sync_channel_id IS NOT NULL"))

    if bound:
        # Their vectors and slots belong to the channels: never drop them silently.
        msg = f"{bound} documents are bound to sync channels; delete or unlink them before downgrading"
        raise RuntimeError(msg)

    replace_active_slot(f"ingestion_key IS NULL AND {SLOT_STATUSES}")
    op.drop_index("ux_documents_sync_file", table_name="documents")
    op.drop_constraint("documents_sync_binding_check", "documents", type_="check")
    op.drop_constraint("documents_sync_channel_id_fkey", "documents", type_="foreignkey")
    op.drop_column("documents", "storage_revision")
    op.drop_column("documents", "storage_version")
    op.drop_column("documents", "sync_path")
    op.drop_column("documents", "sync_channel_id")
    op.drop_index("idx_sync_runs_channel", table_name="sync_runs")
    op.drop_table("sync_runs")
    op.drop_index("idx_sync_channels_space", table_name="sync_channels")
    op.drop_table("sync_channels")
