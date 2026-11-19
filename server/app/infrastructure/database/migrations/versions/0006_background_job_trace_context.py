"""Trace context of the request that submitted a background job.

A separate nullable column, not a dispatch_payload key: relays and task handlers of the
previous release validate the payload strictly and never read this column, so mixed-version
deployments and rollbacks keep working.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "background_jobs",
        sa.Column("trace_context", postgresql.JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("background_jobs", "trace_context")
