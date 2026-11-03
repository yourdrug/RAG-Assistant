"""Preserve benchmark history and request date for dialogue and temporal cases."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_questions", sa.Column("history", postgresql.JSONB, nullable=True))
    op.add_column("benchmark_questions", sa.Column("as_of_date", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("benchmark_questions", "as_of_date")
    op.drop_column("benchmark_questions", "history")
