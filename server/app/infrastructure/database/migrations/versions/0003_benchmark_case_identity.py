"""Preserve v2 case identity and question style across datasets."""

from alembic import op
import sqlalchemy as sa


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name in ("case_id", "variant_of", "question_style"):
        op.add_column("benchmark_questions", sa.Column(name, sa.String(100), nullable=True))


def downgrade() -> None:
    for name in ("question_style", "variant_of", "case_id"):
        op.drop_column("benchmark_questions", name)
