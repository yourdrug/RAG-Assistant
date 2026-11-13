"""Parent scopes of chunks for parent-child retrieval."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_nodes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "creation_date",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "act_version_id",
            sa.Integer(),
            sa.ForeignKey("act_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("node_id", sa.Text(), nullable=False),
        sa.Column("parent_id", sa.Text(), nullable=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("label", sa.String(500), nullable=False),
        sa.Column("number", sa.String(64), nullable=True),
        sa.Column("depth", sa.Integer(), nullable=False),
        sa.Column("chunk_start", sa.Integer(), nullable=False),
        sa.Column("chunk_end", sa.Integer(), nullable=False),
        sa.Column("pages", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("text_hash", sa.String(16), nullable=False),
        sa.UniqueConstraint("document_id", "node_id", name="ux_document_nodes_document_node"),
    )


def downgrade() -> None:
    op.drop_table("document_nodes")
