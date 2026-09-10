"""Merge heads: widen_act_version_date_source + add_document_version_optimistic_locking

Revision ID: c1d2e3f4a5b6
Revises: b3c4d5e6f7a8, b4c5d6e7f8a9
Create Date: 2026-09-10
"""

from collections.abc import Sequence

revision: str = "c1d2e3f4a5b6"
down_revision: str | Sequence[str] | None = ("b3c4d5e6f7a8", "b4c5d6e7f8a9")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
