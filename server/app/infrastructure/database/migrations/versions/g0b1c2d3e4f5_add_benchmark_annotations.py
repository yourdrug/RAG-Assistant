"""Add optional benchmark evidence annotations.

Revision ID: g0b1c2d3e4f5
Revises: f9a0b1c2d3e4
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "g0b1c2d3e4f5"
down_revision = "f9a0b1c2d3e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_questions", sa.Column("annotations", JSONB(), nullable=True))
    op.create_check_constraint(
        "chk_benchmark_questions_annotations_object",
        "benchmark_questions",
        "annotations IS NULL OR jsonb_typeof(annotations) = 'object'",
    )


def downgrade() -> None:
    op.drop_constraint("chk_benchmark_questions_annotations_object", "benchmark_questions", type_="check")
    op.drop_column("benchmark_questions", "annotations")
