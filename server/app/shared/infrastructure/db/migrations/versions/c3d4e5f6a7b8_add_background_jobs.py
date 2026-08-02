"""add background_jobs table for task tracking

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-08-02 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS background_jobs CASCADE")

    op.create_table(
        "background_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("job_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("related_id", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.String(length=32), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed')",
            name="background_jobs_status_check",
        ),
    )

    op.create_index("idx_background_jobs_status", "background_jobs", ["status"])
    op.create_index("idx_background_jobs_job_type", "background_jobs", ["job_type"])
    op.create_index("idx_background_jobs_creation_date", "background_jobs", ["creation_date"])


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS background_jobs")
