"""add chat_logs table for Q&A quality tracking

Revision ID: h3i4j5k6l7m8
Revises: a2b3c4d5e6f7
Create Date: 2026-08-11 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "h3i4j5k6l7m8"
down_revision: str | Sequence[str] | None = "a2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "creation_date",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "conversation_id",
            sa.Integer(),
            sa.ForeignKey("conversations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("breadth", sa.String(20), nullable=True),
        sa.Column("domain", sa.String(20), nullable=True),
        sa.Column("retrieval_count", sa.Integer(), nullable=True),
        sa.Column("reranker_score", sa.Float(), nullable=True),
    )

    op.create_index("idx_chat_logs_user_id", "chat_logs", ["user_id"])
    op.create_index("idx_chat_logs_creation_date", "chat_logs", ["creation_date"])
    op.create_index("idx_chat_logs_domain", "chat_logs", ["domain"])


def downgrade() -> None:
    op.drop_index("idx_chat_logs_domain", table_name="chat_logs")
    op.drop_index("idx_chat_logs_creation_date", table_name="chat_logs")
    op.drop_index("idx_chat_logs_user_id", table_name="chat_logs")
    op.drop_table("chat_logs")
