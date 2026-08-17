"""add condense_enabled and rolling_summary_enabled config parameters

Revision ID: m1n2o3p4q5r6
Revises: k6l7m8n9o0p1
Create Date: 2026-08-17 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "m1n2o3p4q5r6"
down_revision: str | Sequence[str] | None = "k6l7m8n9o0p1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO config_parameters (key, value, value_type, category, description, min_value, max_value) VALUES
        ('condense_enabled',        'true',  'bool', 'rag', 'Rewrite follow-up questions into self-contained queries using chat history', NULL, NULL),
        ('rolling_summary_enabled', 'true',  'bool', 'rag', 'Maintain a rolling summary of long conversations (fire-and-forget background task)', NULL, NULL)
    ON CONFLICT (key) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM config_parameters WHERE key IN (
            'condense_enabled', 'rolling_summary_enabled'
        )
    """)
