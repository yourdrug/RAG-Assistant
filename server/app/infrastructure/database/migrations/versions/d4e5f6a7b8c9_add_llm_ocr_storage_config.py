"""add LLM, OCR, Storage dynamic config parameters

Revision ID: d4e5f6a7b8c9
Revises: 8e19a1cfb758
Create Date: 2026-08-06 12:00:00.000000

"""

from collections.abc import Sequence


revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "8e19a1cfb758"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Default values are seeded at startup from settings (initialization.py)
    pass


def downgrade() -> None:
    pass
