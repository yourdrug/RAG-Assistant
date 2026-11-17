"""Replace legacy quality logs with answer attempts and config snapshots.

Existing logs cannot be backfilled truthfully; this migration intentionally deletes them.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:  # noqa: PLR0915 - explicit schema operations for one atomic migration.
    op.execute("DELETE FROM chat_logs")
    op.create_table(
        "rag_config_snapshots",
        sa.Column("hash", sa.String(64), primary_key=True),
        sa.Column("data", postgresql.JSONB, nullable=False),
        sa.Column("creation_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.drop_column("chat_logs", "retrieval_count")
    op.drop_column("chat_logs", "reranker_score")
    op.add_column("chat_logs", sa.Column("message_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_chat_logs_message_id", "chat_logs", "messages", ["message_id"], ["id"], ondelete="SET NULL"
    )
    op.add_column("chat_logs", sa.Column("space_id", sa.Integer(), nullable=False))
    op.create_foreign_key(
        "fk_chat_logs_space_id", "chat_logs", "knowledge_spaces", ["space_id"], ["id"], ondelete="RESTRICT"
    )
    op.add_column("chat_logs", sa.Column("pack_key", sa.String(100), nullable=True))
    op.add_column("chat_logs", sa.Column("request_id", sa.String(100), nullable=True))
    op.add_column("chat_logs", sa.Column("depth_requested", sa.String(20), nullable=True))
    op.add_column("chat_logs", sa.Column("as_of_date", sa.Date(), nullable=True))
    op.add_column("chat_logs", sa.Column("outcome", sa.String(30), nullable=False))
    op.add_column("chat_logs", sa.Column("error_code", sa.String(100), nullable=True))
    op.add_column("chat_logs", sa.Column("condensed_query", sa.Text(), nullable=True))
    op.add_column("chat_logs", sa.Column("sub_queries", postgresql.JSONB, nullable=True))
    op.add_column("chat_logs", sa.Column("refined_query", sa.Text(), nullable=True))
    op.add_column("chat_logs", sa.Column("confidence", sa.Float(), nullable=True))
    op.add_column("chat_logs", sa.Column("sources_count", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("candidates_count", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("top_rerank_score", sa.Float(), nullable=True))
    op.add_column("chat_logs", sa.Column("aux_tokens", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("ttft_ms", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("retrieval_ms", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("rerank_ms", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("generation_ms", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("config_hash", sa.String(64), nullable=True))
    op.create_foreign_key(
        "fk_chat_logs_config_hash",
        "chat_logs",
        "rag_config_snapshots",
        ["config_hash"],
        ["hash"],
        ondelete="RESTRICT",
    )
    op.add_column("chat_logs", sa.Column("config_version", sa.String(100), nullable=True))
    op.add_column("chat_logs", sa.Column("evidence_profile", sa.String(200), nullable=True))
    op.create_index("idx_chat_logs_outcome", "chat_logs", ["outcome"])
    op.create_index("idx_chat_logs_message_id", "chat_logs", ["message_id"], unique=True)
    op.create_index("idx_chat_logs_config_hash", "chat_logs", ["config_hash"])
    op.create_index("idx_chat_logs_space_date", "chat_logs", ["space_id", "creation_date"])
    op.create_index("idx_chat_logs_date_outcome", "chat_logs", ["creation_date", "outcome"])


def downgrade() -> None:  # noqa: PLR0915 - reverse the schema operations explicitly.
    op.execute("DELETE FROM chat_logs")
    op.drop_index("idx_chat_logs_outcome", table_name="chat_logs")
    op.drop_index("idx_chat_logs_message_id", table_name="chat_logs")
    op.drop_index("idx_chat_logs_config_hash", table_name="chat_logs")
    op.drop_index("idx_chat_logs_space_date", table_name="chat_logs")
    op.drop_index("idx_chat_logs_date_outcome", table_name="chat_logs")
    op.drop_column("chat_logs", "evidence_profile")
    op.drop_column("chat_logs", "config_version")
    op.drop_constraint("fk_chat_logs_config_hash", "chat_logs", type_="foreignkey")
    op.drop_column("chat_logs", "config_hash")
    op.drop_column("chat_logs", "generation_ms")
    op.drop_column("chat_logs", "rerank_ms")
    op.drop_column("chat_logs", "retrieval_ms")
    op.drop_column("chat_logs", "ttft_ms")
    op.drop_column("chat_logs", "aux_tokens")
    op.drop_column("chat_logs", "top_rerank_score")
    op.drop_column("chat_logs", "candidates_count")
    op.drop_column("chat_logs", "sources_count")
    op.drop_column("chat_logs", "confidence")
    op.drop_column("chat_logs", "refined_query")
    op.drop_column("chat_logs", "sub_queries")
    op.drop_column("chat_logs", "condensed_query")
    op.drop_column("chat_logs", "error_code")
    op.drop_column("chat_logs", "outcome")
    op.drop_column("chat_logs", "as_of_date")
    op.drop_column("chat_logs", "depth_requested")
    op.drop_column("chat_logs", "request_id")
    op.drop_column("chat_logs", "pack_key")
    op.drop_constraint("fk_chat_logs_space_id", "chat_logs", type_="foreignkey")
    op.drop_column("chat_logs", "space_id")
    op.drop_constraint("fk_chat_logs_message_id", "chat_logs", type_="foreignkey")
    op.drop_column("chat_logs", "message_id")
    op.add_column("chat_logs", sa.Column("retrieval_count", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("reranker_score", sa.Float(), nullable=True))
    op.drop_table("rag_config_snapshots")
