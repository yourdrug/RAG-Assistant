"""add doc_domain column to documents and chunks

Revision ID: a2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-08-11 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("doc_domain", sa.String(16), nullable=False, server_default="general"),
    )
    op.add_column(
        "chunks",
        sa.Column("doc_domain", sa.String(16), nullable=False, server_default="general"),
    )
    op.create_check_constraint(
        "documents_doc_domain_check",
        "documents",
        "doc_domain IN ('legal', 'general')",
    )
    op.create_check_constraint(
        "chunks_doc_domain_check",
        "chunks",
        "doc_domain IN ('legal', 'general')",
    )
    op.create_index("idx_documents_doc_domain", "documents", ["doc_domain"])


def downgrade() -> None:
    op.drop_index("idx_documents_doc_domain", table_name="documents")
    op.drop_constraint("chunks_doc_domain_check", "chunks")
    op.drop_constraint("documents_doc_domain_check", "documents")
    op.drop_column("chunks", "doc_domain")
    op.drop_column("documents", "doc_domain")
