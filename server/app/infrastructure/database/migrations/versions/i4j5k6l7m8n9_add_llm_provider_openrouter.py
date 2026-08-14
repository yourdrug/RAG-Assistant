"""add LLM provider selection and OpenRouter config parameters

Revision ID: i4j5k6l7m8n9
Revises: h3i4j5k6l7m8
Create Date: 2026-08-14 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "i4j5k6l7m8n9"
down_revision: str | Sequence[str] | None = "h3i4j5k6l7m8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO config_parameters (key, value, value_type, category, description, min_value, max_value, allowed_values) VALUES
        -- LLM Provider selection
        ('llm_provider',        'ollama',      'str', 'llm', 'LLM provider: ollama | openrouter', NULL, NULL, '{"values": ["ollama", "openrouter"]}'),
        -- OpenRouter
        ('openrouter_model',    'qwen/qwen-2.5-7b-instruct', 'str', 'openrouter', 'OpenRouter model ID', NULL, NULL, NULL)
    ON CONFLICT (key) DO NOTHING
    """)

    # Update existing llm_model description to indicate it's for Ollama
    op.execute("""
        UPDATE config_parameters
        SET description = 'LLM model name (Ollama only, used when llm_provider=ollama)'
        WHERE key = 'llm_model'
    """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM config_parameters WHERE key IN ('llm_provider', 'openrouter_model')
    """)
    op.execute("""
        UPDATE config_parameters
        SET description = 'LLM model name (Ollama)'
        WHERE key = 'llm_model'
    """)
