"""Staged document revisions: re-processing keeps the published chunks searchable.

Also admits the outbox operations ``publish_revision`` (new) and ``rename_document``,
which the operation check omitted although document renames enqueue it.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


OLD_OPERATIONS = (
    "'upsert_chunks', 'delete_by_document', 'delete_chunks', 'update_metadata', 'set_document_id'"
)
NEW_OPERATIONS = f"{OLD_OPERATIONS}, 'rename_document', 'publish_revision'"


def replace_operation_check(operations: str) -> None:
    op.drop_constraint("vector_store_outbox_operation_check", "vector_store_outbox", type_="check")
    op.create_check_constraint(
        "vector_store_outbox_operation_check", "vector_store_outbox", f"operation IN ({operations})"
    )


def upgrade() -> None:
    replace_operation_check(NEW_OPERATIONS)
    op.create_table(
        "document_staged_revisions",
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("token", sa.String(length=32), nullable=False),
        sa.Column("nodes", postgresql.JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="document_staged_revisions_document_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_id", name="document_staged_revisions_pkey"),
    )
    op.create_table(
        "chunk_staging",
        # Same sequence as chunks: a point keeps its id when the revision is published.
        sa.Column(
            "id", sa.Integer(), server_default=sa.text("nextval('chunks_id_seq'::regclass)"), nullable=False
        ),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embed_text", sa.Text(), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("visibility", sa.String(length=20), nullable=False),
        sa.Column("doc_domain", sa.String(length=32), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=True),
        sa.Column("group_id", sa.Integer(), nullable=True),
        sa.Column("space_id", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=16), nullable=True),
        sa.Column("domain_metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "context_metadata", postgresql.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("pack_metadata", postgresql.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("heading", sa.String(length=500), nullable=True),
        sa.Column("heading_level", sa.Integer(), nullable=True),
        sa.Column("content_type", sa.String(length=32), nullable=True),
        sa.Column("doc_title", sa.String(length=500), nullable=True),
        sa.Column("doc_type", sa.String(length=64), nullable=True),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.CheckConstraint(
            "jsonb_typeof(context_metadata) = 'object'", name="chk_chunk_staging_context_metadata_object"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(pack_metadata) = 'object'", name="chk_chunk_staging_pack_metadata_object"
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["document_staged_revisions.document_id"],
            name="chunk_staging_document_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="chunk_staging_owner_id_fkey", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["group_id"], ["groups.id"], name="chunk_staging_group_id_fkey", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["space_id"], ["knowledge_spaces.id"], name="chunk_staging_space_id_fkey", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="chunk_staging_pkey"),
        sa.UniqueConstraint("document_id", "chunk_index", name="ux_chunk_staging_document_index"),
    )


def downgrade() -> None:
    op.execute("DELETE FROM vector_store_outbox WHERE operation IN ('rename_document', 'publish_revision')")
    replace_operation_check(OLD_OPERATIONS)
    op.drop_table("chunk_staging")
    op.drop_table("document_staged_revisions")
