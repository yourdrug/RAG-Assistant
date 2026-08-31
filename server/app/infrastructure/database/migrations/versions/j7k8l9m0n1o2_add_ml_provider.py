"""add ML provider selection (tei | deepinfra) config parameter

Revision ID: j7k8l9m0n1o2
Revises: i4j5k6l7m8n9
Create Date: 2026-08-31 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "j7k8l9m0n1o2"
down_revision: str | Sequence[str] | None = "q6r7s8t9u0v1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO config_parameters (key, value, value_type, category, description, min_value, max_value, allowed_values) VALUES
        ('ml_provider',         'tei',  'str', 'ml', 'ML provider for embeddings+reranking: tei | deepinfra', NULL, NULL, '{"values": ["tei", "deepinfra"]}')
    ON CONFLICT (key) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM config_parameters WHERE key = 'ml_provider'")
