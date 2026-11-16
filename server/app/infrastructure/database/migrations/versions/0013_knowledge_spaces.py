"""Knowledge spaces: a space per document, chunk and conversation.

Creates the starting space on the neutral pack and assigns every existing document,
chunk and conversation to it. Vector payloads gain ``metadata.space_id`` only on
reindexing: reindex every document from scratch after upgrading.
"""

from alembic import op
import sqlalchemy as sa


revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

SLOT_STATUSES = "'pending', 'processing', 'indexing', 'done', 'failed'"


def upgrade() -> None:
    op.create_table(
        "knowledge_spaces",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("pack_key", sa.String(64), nullable=False),
    )
    op.create_table(
        "knowledge_space_upload_groups",
        sa.Column(
            "space_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_spaces.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("group_id", sa.Integer(), sa.ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True),
    )
    starting = op.get_bind().execute(
        sa.text(
            "INSERT INTO knowledge_spaces (creation_date, name, pack_key) "
            "VALUES (NOW(), 'Общее', 'generic') RETURNING id"
        )
    )
    space_id = starting.scalar_one()

    for table in ("documents", "chunks"):
        op.add_column(table, sa.Column("space_id", sa.Integer(), nullable=True))
        op.execute(sa.text(f"UPDATE {table} SET space_id = :space_id").bindparams(space_id=space_id))  # noqa: S608
        op.alter_column(table, "space_id", nullable=False)
        op.create_foreign_key(
            f"fk_{table}_space_id", table, "knowledge_spaces", ["space_id"], ["id"], ondelete="RESTRICT"
        )

    op.create_index("idx_documents_space", "documents", ["space_id"])
    op.create_index("ix_chunks_space_id", "chunks", ["space_id"])

    # Preserve existing timelines under the same identity used for new editions.
    op.execute(
        sa.text("UPDATE regulatory_acts SET visibility_scope = :prefix || visibility_scope").bindparams(
            prefix=f"space:{space_id}:"
        )
    )

    op.add_column(
        "conversations",
        sa.Column(
            "space_id", sa.Integer(), sa.ForeignKey("knowledge_spaces.id", ondelete="CASCADE"), nullable=True
        ),
    )
    op.execute(sa.text("UPDATE conversations SET space_id = :space_id").bindparams(space_id=space_id))

    # A filename slot is per space: the same file may be loaded into two spaces.
    op.drop_index("ux_documents_active_slot", table_name="documents")
    op.drop_index("ux_documents_ingestion_slot", table_name="documents")
    op.create_index(
        "ux_documents_active_slot",
        "documents",
        [
            "space_id",
            sa.literal_column("COALESCE(owner_id, 0)"),
            sa.literal_column("COALESCE(group_id, 0)"),
            "filename",
        ],
        unique=True,
        postgresql_where=sa.text(f"ingestion_key IS NULL AND status IN ({SLOT_STATUSES})"),
    )
    op.create_index(
        "ux_documents_ingestion_slot",
        "documents",
        [
            "space_id",
            "ingestion_key",
            sa.literal_column("COALESCE(owner_id, 0)"),
            sa.literal_column("COALESCE(group_id, 0)"),
        ],
        unique=True,
        postgresql_where=sa.text(f"ingestion_key IS NOT NULL AND status IN ({SLOT_STATUSES})"),
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE regulatory_acts "
            "SET visibility_scope = regexp_replace(visibility_scope, '^space:[0-9]+:', '')"
        )
    )
    op.drop_index("ux_documents_ingestion_slot", table_name="documents")
    op.drop_index("ux_documents_active_slot", table_name="documents")
    op.create_index(
        "ux_documents_active_slot",
        "documents",
        [sa.literal_column("COALESCE(owner_id, 0)"), sa.literal_column("COALESCE(group_id, 0)"), "filename"],
        unique=True,
        postgresql_where=sa.text(f"ingestion_key IS NULL AND status IN ({SLOT_STATUSES})"),
    )
    op.create_index(
        "ux_documents_ingestion_slot",
        "documents",
        [
            "ingestion_key",
            sa.literal_column("COALESCE(owner_id, 0)"),
            sa.literal_column("COALESCE(group_id, 0)"),
        ],
        unique=True,
        postgresql_where=sa.text(f"ingestion_key IS NOT NULL AND status IN ({SLOT_STATUSES})"),
    )
    op.drop_column("conversations", "space_id")
    op.drop_index("ix_chunks_space_id", table_name="chunks")
    op.drop_index("idx_documents_space", table_name="documents")

    for table in ("chunks", "documents"):
        op.drop_constraint(f"fk_{table}_space_id", table, type_="foreignkey")
        op.drop_column(table, "space_id")

    op.drop_table("knowledge_space_upload_groups")
    op.drop_table("knowledge_spaces")
