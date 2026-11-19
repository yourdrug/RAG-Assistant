"""Recorded LLM responses for benchmark replay and the replay mode of a sweep."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_replay_entries",
        sa.Column("key", sa.VARCHAR(length=64), autoincrement=False, nullable=False),
        sa.Column("model", sa.VARCHAR(length=255), autoincrement=False, nullable=True),
        sa.Column("status_code", sa.SMALLINT(), autoincrement=False, nullable=False),
        sa.Column("headers", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("body", postgresql.BYTEA(), autoincrement=False, nullable=False),
        sa.Column("input_tokens", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("output_tokens", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "last_used_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key", name="llm_replay_entries_pkey"),
    )
    op.create_index("idx_llm_replay_entries_last_used_at", "llm_replay_entries", ["last_used_at"])
    op.add_column(
        "benchmark_sweeps",
        sa.Column(
            "llm_replay_mode",
            sa.VARCHAR(length=20),
            server_default=sa.text("'off'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "benchmark_sweeps_llm_replay_mode_check",
        "benchmark_sweeps",
        "llm_replay_mode IN ('off', 'record', 'replay_only', 'refresh')",
    )


def downgrade() -> None:
    op.drop_constraint("benchmark_sweeps_llm_replay_mode_check", "benchmark_sweeps", type_="check")
    op.drop_column("benchmark_sweeps", "llm_replay_mode")
    op.drop_index("idx_llm_replay_entries_last_used_at", table_name="llm_replay_entries")
    op.drop_table("llm_replay_entries")
