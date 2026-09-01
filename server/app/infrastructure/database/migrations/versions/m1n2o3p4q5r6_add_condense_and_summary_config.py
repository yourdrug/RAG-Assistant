"""add condense_enabled and rolling_summary_enabled config parameters

Revision ID: m1n2o3p4q5r6
Revises: k6l7m8n9o0p1
Create Date: 2026-08-17 12:00:00.000000

"""

from collections.abc import Sequence


revision: str = "m1n2o3p4q5r6"
down_revision: str | Sequence[str] | None = "k6l7m8n9o0p1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Default values are seeded at startup from settings (initialization.py)
    pass


def downgrade() -> None:
    pass
