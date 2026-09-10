"""add structural metadata columns to chunks

Revision ID: d7e8f9a0b1c2
Revises: 08a6c8d44aea
Create Date: 2026-09-10 14:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d7e8f9a0b1c2"
down_revision: Union[str, Sequence[str], None] = "08a6c8d44aea"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("section", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("heading", sa.String(500), nullable=True))
    op.add_column("chunks", sa.Column("heading_level", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("content_type", sa.String(32), nullable=True))
    op.add_column("chunks", sa.Column("doc_title", sa.String(500), nullable=True))
    op.add_column("chunks", sa.Column("doc_type", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "doc_type")
    op.drop_column("chunks", "doc_title")
    op.drop_column("chunks", "content_type")
    op.drop_column("chunks", "heading_level")
    op.drop_column("chunks", "heading")
    op.drop_column("chunks", "section")
