"""Move act edition fields out of the core chunk schema into pack-owned metadata.

Destructive: chunk edition columns are dropped without carrying their values over.
Reindex every document from scratch after upgrading; the law pack rewrites
``chunks.pack_metadata`` and the vector payload from ``act_versions``.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("idx_chunks_act_version_id", table_name="chunks")

    for column in ("act_version_id", "effective_from", "effective_to", "is_current"):
        op.drop_column("chunks", column)

    op.add_column(
        "chunks",
        sa.Column(
            "pack_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_check_constraint(
        "chk_chunks_pack_metadata_object", "chunks", "jsonb_typeof(pack_metadata) = 'object'"
    )
    op.drop_column("document_nodes", "act_version_id")
    # Old entries address points by act_version_id, which payloads no longer carry.
    op.execute(sa.text("DELETE FROM vector_store_outbox WHERE operation = 'update_metadata'"))

    # A document is exactly one edition; keep the newest row of any duplicate.
    op.execute(
        sa.text(
            "DELETE FROM act_versions AS older USING act_versions AS newer "
            "WHERE older.document_id = newer.document_id AND older.id < newer.id"
        )
    )
    op.create_unique_constraint("ux_act_versions_document_id", "act_versions", ["document_id"])


def downgrade() -> None:
    op.drop_constraint("ux_act_versions_document_id", "act_versions", type_="unique")
    op.add_column(
        "document_nodes",
        sa.Column(
            "act_version_id",
            sa.Integer(),
            sa.ForeignKey("act_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.drop_constraint("chk_chunks_pack_metadata_object", "chunks", type_="check")
    op.drop_column("chunks", "pack_metadata")
    op.add_column(
        "chunks", sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.text("true"))
    )
    op.add_column("chunks", sa.Column("effective_to", sa.Date(), nullable=True))
    op.add_column("chunks", sa.Column("effective_from", sa.Date(), nullable=True))
    op.add_column(
        "chunks",
        sa.Column(
            "act_version_id",
            sa.Integer(),
            sa.ForeignKey("act_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "idx_chunks_act_version_id",
        "chunks",
        ["act_version_id"],
        postgresql_where=sa.text("act_version_id IS NOT NULL"),
    )
