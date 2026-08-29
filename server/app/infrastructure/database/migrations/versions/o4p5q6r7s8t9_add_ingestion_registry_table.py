"""Add ingestion_registry table — replaces JSON-file based registry

Revision ID: o4p5q6r7s8t9
Revises: n3o4p5q6r7s8
Create Date: 2026-08-29
"""

from alembic import op
import sqlalchemy as sa


revision = "o4p5q6r7s8t9"
down_revision = "f5f0f1a39ba1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingestion_registry",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("creation_date", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("file_hash", sa.String(255), nullable=False, server_default=""),
        sa.Column("source", sa.String(1000), nullable=False, server_default=""),
        sa.Column("chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("indexed_at", sa.String(50), nullable=True),
    )
    op.create_index(
        "idx_ingestion_registry_filename",
        "ingestion_registry",
        ["filename"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("idx_ingestion_registry_filename", table_name="ingestion_registry")
    op.drop_table("ingestion_registry")
