"""add quality_score to documents and ocr_min_chars to config_parameters

Revision ID: l2m3n4o5p6q7
Revises: 69930d1c204f
Create Date: 2026-08-17 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "l2m3n4o5p6q7"
down_revision: str | Sequence[str] | None = "m1n2o3p4q5r6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # documents: add quality_score (bad_ratio from PDF quality assessment)
    op.add_column(
        "documents",
        sa.Column("quality_score", sa.Float(), nullable=True),
    )

    # Default values are seeded at startup from settings (initialization.py)


def downgrade() -> None:
    op.drop_column("documents", "quality_score")
