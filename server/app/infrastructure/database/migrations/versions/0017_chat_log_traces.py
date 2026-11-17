"""Store bounded request traces separately from the answer journal."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_log_traces",
        sa.Column(
            "chat_log_id", sa.Integer(), sa.ForeignKey("chat_logs.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("data", postgresql.JSONB, nullable=False),
        sa.CheckConstraint("octet_length(data::text) <= 65536", name="ck_chat_log_trace_size"),
    )


def downgrade() -> None:
    op.drop_table("chat_log_traces")
