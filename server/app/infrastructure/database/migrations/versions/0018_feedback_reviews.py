"""User reactions to answers and administrative reviews of journal entries."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "message_feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "message_id", sa.Integer(), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rating", sa.String(8), nullable=False),
        sa.Column("reasons", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("creation_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("update_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("message_id", "user_id", name="uq_message_feedback_message_user"),
        sa.CheckConstraint("rating IN ('up', 'down')", name="ck_message_feedback_rating"),
        sa.CheckConstraint(
            "rating = 'down' OR reasons = '[]'::jsonb", name="ck_message_feedback_reasons_down"
        ),
    )
    op.create_table(
        "chat_log_reviews",
        sa.Column(
            "chat_log_id", sa.Integer(), sa.ForeignKey("chat_logs.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("reviewer_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column(
            "benchmark_question_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_questions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("update_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('new', 'benchmark_candidate', 'in_benchmark', 'content_gap', "
            "'pipeline_issue', 'wont_fix')",
            name="ck_chat_log_reviews_status",
        ),
    )
    op.create_index("idx_chat_log_reviews_status", "chat_log_reviews", ["status"])


def downgrade() -> None:
    op.drop_index("idx_chat_log_reviews_status", table_name="chat_log_reviews")
    op.drop_table("chat_log_reviews")
    op.drop_table("message_feedback")
