"""Add background_jobs.heartbeat_at for heartbeat-based orphan recovery.

Long arq jobs touch heartbeat_at periodically while running. The orphan
reaper only fails a 'running' job whose last heartbeat is older than the
timeout — a legitimately long-running job no longer gets marked failed.

Revision ID: y6b7c8d9e0f1
Revises: x1a2b3c4d5e6
Create Date: 2026-09-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "y6b7c8d9e0f1"
down_revision: str | Sequence[str] | None = "x1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("background_jobs", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("idx_background_jobs_heartbeat", "background_jobs", ["heartbeat_at"])


def downgrade() -> None:
    op.drop_index("idx_background_jobs_heartbeat", table_name="background_jobs")
    op.drop_column("background_jobs", "heartbeat_at")
