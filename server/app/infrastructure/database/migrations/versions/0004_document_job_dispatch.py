"""Persist document processing submissions with their documents."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "background_jobs",
        sa.Column("dispatch_payload", postgresql.JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("background_jobs", "dispatch_payload")
