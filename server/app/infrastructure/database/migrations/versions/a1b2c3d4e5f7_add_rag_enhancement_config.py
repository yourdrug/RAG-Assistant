"""add relevance_gate, decomposition, cache config parameters

Revision ID: a1b2c3d4e5f7
Revises: f1a2b3c4d5e6
Create Date: 2026-08-11 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1b2c3d4e5f7"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO config_parameters (key, value, value_type, category, description, min_value, max_value) VALUES
        ('relevance_gate_enabled', 'false', 'bool',  'rag', 'Enable semantic relevance check before LLM generation (Self-RAG-lite)', NULL, NULL),
        ('decomposition_enabled',  'false', 'bool',  'rag', 'Enable automatic decomposition of compound questions into sub-queries', NULL, NULL),
        ('cache_enabled',          'false', 'bool',  'rag', 'Enable semantic answer cache (requires security review before enabling in prod)', NULL, NULL)
    ON CONFLICT (key) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM config_parameters WHERE key IN (
            'relevance_gate_enabled', 'decomposition_enabled', 'cache_enabled'
        )
    """)
