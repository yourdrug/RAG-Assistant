"""Add domain profile registry.

config_parameters domain_key, remove doc_domain CHECK constraints, add
regulatory_acts/act_versions tables, add versioning and domain_metadata
columns to chunks.

Revision ID: x1a2b3c4d5e6
Revises: w5x6y7z8a9b0
Create Date: 2026-09-04
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "x1a2b3c4d5e6"
down_revision = "w5x6y7z8a9b0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. config_parameters: add domain_key, replace unique constraint
    op.add_column("config_parameters", sa.Column("domain_key", sa.String(32), nullable=True))
    op.drop_constraint("config_parameters_key_key", "config_parameters", type_="unique")
    op.create_unique_constraint("ux_config_parameters_key_domain", "config_parameters", ["key", "domain_key"])
    op.create_index(
        "idx_config_parameters_domain_key",
        "config_parameters",
        ["domain_key"],
        postgresql_where="domain_key IS NOT NULL",
    )

    # 2. documents: remove doc_domain CHECK constraint, widen column
    op.drop_constraint("documents_doc_domain_check", "documents", type_="check")
    op.alter_column("documents", "doc_domain", type_=sa.String(32), existing_server_default="general")

    # 3. chunks: remove doc_domain CHECK constraint, widen column, add new columns
    op.drop_constraint("chunks_doc_domain_check", "chunks", type_="check")
    op.alter_column("chunks", "doc_domain", type_=sa.String(32), existing_server_default="general")
    op.add_column("chunks", sa.Column("domain_metadata", JSONB, nullable=True))
    op.add_column("chunks", sa.Column("effective_from", sa.Date(), nullable=True))
    op.add_column("chunks", sa.Column("effective_to", sa.Date(), nullable=True))
    op.add_column("chunks", sa.Column("is_current", sa.Boolean(), nullable=False, server_default="true"))

    # 4. regulatory_acts table
    op.create_table(
        "regulatory_acts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("creation_date", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("act_type", sa.String(32), nullable=False),
        sa.Column("act_number", sa.String(100), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("issuing_authority", sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("act_type", "act_number", name="ux_regulatory_acts_type_number"),
    )

    # 5. act_versions table
    op.create_table(
        "act_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("creation_date", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "act_id", sa.Integer(), sa.ForeignKey("regulatory_acts.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column(
            "document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("effective_from", sa.Date(), nullable=True),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("date_source", sa.String(16), nullable=False, server_default="extracted"),
        sa.Column("date_confidence", sa.Float(), nullable=True),
        sa.Column("verified_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_act_versions_act_id", "act_versions", ["act_id"])
    # Guarantee: an act cannot have two "current" versions at once
    op.create_index(
        "ux_act_versions_one_current",
        "act_versions",
        ["act_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )

    # 6. chunks: add act_version_id FK + index
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
        postgresql_where="act_version_id IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_index("idx_chunks_act_version_id", table_name="chunks")
    op.drop_column("chunks", "act_version_id")
    op.drop_index("ux_act_versions_one_current", table_name="act_versions")
    op.drop_index("idx_act_versions_act_id", table_name="act_versions")
    op.drop_table("act_versions")
    op.drop_table("regulatory_acts")
    op.drop_column("chunks", "is_current")
    op.drop_column("chunks", "effective_to")
    op.drop_column("chunks", "effective_from")
    op.drop_column("chunks", "domain_metadata")
    op.alter_column("chunks", "doc_domain", type_=sa.String(16), existing_server_default="general")
    op.create_check_constraint("chunks_doc_domain_check", "chunks", "doc_domain IN ('legal', 'general')")
    op.alter_column("documents", "doc_domain", type_=sa.String(16), existing_server_default="general")
    op.create_check_constraint(
        "documents_doc_domain_check", "documents", "doc_domain IN ('legal', 'general')"
    )
    op.drop_index("idx_config_parameters_domain_key", table_name="config_parameters")
    op.drop_constraint("ux_config_parameters_key_domain", "config_parameters", type_="unique")
    op.drop_column("config_parameters", "domain_key")
    op.create_unique_constraint("config_parameters_key_key", "config_parameters", ["key"])
