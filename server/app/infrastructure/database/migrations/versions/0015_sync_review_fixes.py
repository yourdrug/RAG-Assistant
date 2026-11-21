"""Sync channel hardening.

- ``document_staged_revisions.source_version``/``source_revision``: the storage version a
  staged revision was made from; a channel document records it only when published.
- ``idx_background_jobs_type_related``: finds the worker job of a sync run.
- ``sync_runs.tallies``: counts of unchanged and excluded files, no longer stored per path.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_staged_revisions", sa.Column("source_version", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "document_staged_revisions", sa.Column("source_revision", sa.String(length=64), nullable=True)
    )
    op.create_index("idx_background_jobs_type_related", "background_jobs", ["job_type", "related_id"])
    op.add_column(
        "sync_runs",
        sa.Column("tallies", postgresql.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
    )
    op.create_check_constraint("sync_runs_tallies_object", "sync_runs", "jsonb_typeof(tallies) = 'object'")


def downgrade() -> None:
    op.drop_constraint("sync_runs_tallies_object", "sync_runs", type_="check")
    op.drop_column("sync_runs", "tallies")
    op.drop_index("idx_background_jobs_type_related", table_name="background_jobs")
    op.drop_column("document_staged_revisions", "source_revision")
    op.drop_column("document_staged_revisions", "source_version")
