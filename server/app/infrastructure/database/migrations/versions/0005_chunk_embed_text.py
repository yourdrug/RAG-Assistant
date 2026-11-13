"""Separate dense embedding text from source content."""

from alembic import op
import sqlalchemy as sa


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embed_text", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "embed_text")
