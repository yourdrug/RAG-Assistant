"""add LLM provider selection and OpenRouter config parameters

Revision ID: i4j5k6l7m8n9
Revises: h3i4j5k6l7m8
Create Date: 2026-08-14 12:00:00.000000

"""

from collections.abc import Sequence


revision: str = "i4j5k6l7m8n9"
down_revision: str | Sequence[str] | None = "h3i4j5k6l7m8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Default values are seeded at startup from settings (initialization.py)
    pass


def downgrade() -> None:
    pass
