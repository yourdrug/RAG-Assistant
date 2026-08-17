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

    # config_parameters: add ocr_min_chars for threshold-based OCR triggering
    op.execute(
        """
        INSERT INTO config_parameters (key, value, value_type, category, description, min_value, max_value)
        VALUES ('ocr_min_chars', '40', 'int', 'ocr',
                'Minimum characters for text layer to be considered valid (pages below this threshold also get OCR)',
                0, 500)
        ON CONFLICT (key) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DELETE FROM config_parameters WHERE key = 'ocr_min_chars'")
    op.drop_column("documents", "quality_score")
