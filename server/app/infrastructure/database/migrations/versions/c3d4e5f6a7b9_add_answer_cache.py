"""add answer_cache table for semantic answer caching

Revision ID: c3d4e5f6a7b9
Revises: b2c3d4e5f6a8
Create Date: 2026-08-11 13:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b9"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "answer_cache",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("creation_date", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("question_embedding_hash", sa.String(64), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("visibility_scope_hash", sa.String(64), nullable=False),
        sa.Column("document_ids", sa.JSON(), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("idx_answer_cache_embedding_hash", "answer_cache", ["question_embedding_hash"])
    op.create_index("idx_answer_cache_visibility", "answer_cache", ["visibility_scope_hash"])


def downgrade() -> None:
    op.drop_index("idx_answer_cache_visibility", table_name="answer_cache")
    op.drop_index("idx_answer_cache_embedding_hash", table_name="answer_cache")
    op.drop_table("answer_cache")
