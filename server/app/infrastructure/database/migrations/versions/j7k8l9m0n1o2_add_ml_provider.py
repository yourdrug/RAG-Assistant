"""add ML provider selection (tei | deepinfra) config parameter

Revision ID: j7k8l9m0n1o2
Revises: i4j5k6l7m8n9
Create Date: 2026-08-31 12:00:00.000000

"""

from collections.abc import Sequence


revision: str = "j7k8l9m0n1o2"
down_revision: str | Sequence[str] | None = "q6r7s8t9u0v1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Default values are seeded at startup from settings (initialization.py)
    pass


def downgrade() -> None:
    pass
