"""add relevance_gate, decomposition, cache config parameters

Revision ID: a1b2c3d4e5f7
Revises: f1a2b3c4d5e6
Create Date: 2026-08-11 12:00:00.000000

"""

from collections.abc import Sequence


revision: str = "a1b2c3d4e5f7"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Default values are seeded at startup from settings (initialization.py)
    pass


def downgrade() -> None:
    pass
