"""Add version_group_id to documents for version chain tracking.

Links documents into version chains without relying on filename matching.
The version_group_id points to the anchor document (first in the chain),
establishing identity independent of filename.
"""

from alembic import op
import sqlalchemy as sa

revision = "a3b4c5d6e7f8"
down_revision = "z8d9e0f1g2h3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("version_group_id", sa.Integer(), nullable=True))
    op.create_index(
        "idx_documents_version_group",
        "documents",
        ["version_group_id"],
        postgresql_where="version_group_id IS NOT NULL",
    )
    op.create_foreign_key(
        "fk_documents_version_group",
        "documents",
        "documents",
        ["version_group_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_documents_version_group", "documents", type_="foreignkey")
    op.drop_index("idx_documents_version_group", postgresql_where="version_group_id IS NOT NULL")
    op.drop_column("documents", "version_group_id")
