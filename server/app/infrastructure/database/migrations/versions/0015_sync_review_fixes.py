"""Sync channel hardening.

- ``document_staged_revisions.source_version``/``source_revision``: the storage version a
  staged revision was made from; a channel document records it only when published.
- ``idx_background_jobs_type_related``: finds the worker job of a sync run.
"""

from alembic import op
import sqlalchemy as sa


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


def downgrade() -> None:
    op.drop_index("idx_background_jobs_type_related", table_name="background_jobs")
    op.drop_column("document_staged_revisions", "source_revision")
    op.drop_column("document_staged_revisions", "source_version")
