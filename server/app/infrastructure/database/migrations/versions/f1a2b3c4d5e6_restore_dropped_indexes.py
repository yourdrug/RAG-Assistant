"""restore indexes dropped by 8e19a1cfb758

Revision ID: f1a2b3c4d5e6
Revises: e5f6a7b8c9d0
Create Date: 2026-08-09 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Restore the partial unique index that enforces at most one active document
    # per (owner_id, filename) — the app's concurrency-safety mechanism for uploads.
    op.create_index(
        "ux_documents_active_slot",
        "documents",
        ["owner_id", "filename"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing', 'done')"),
    )

    # Documents
    op.create_index("idx_documents_owner", "documents", ["owner_id"], unique=False)
    op.create_index("idx_documents_group", "documents", ["group_id"], unique=False)
    op.create_index("idx_documents_visibility", "documents", ["visibility"], unique=False)
    op.create_index("idx_documents_status", "documents", ["status"], unique=False)

    # Conversations
    op.create_index("idx_conversations_user", "conversations", ["user_id"], unique=False)

    # Messages
    op.create_index("idx_messages_conv", "messages", ["conversation_id"], unique=False)

    # API keys
    op.create_index("idx_api_keys_user_id", "api_keys", ["user_id"], unique=False)

    # Config parameters
    op.create_index("idx_config_parameters_category", "config_parameters", ["category"], unique=False)

    # Background jobs
    op.create_index("idx_background_jobs_status", "background_jobs", ["status"], unique=False)
    op.create_index("idx_background_jobs_job_type", "background_jobs", ["job_type"], unique=False)
    op.create_index("idx_background_jobs_creation_date", "background_jobs", ["creation_date"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_background_jobs_creation_date", table_name="background_jobs")
    op.drop_index("idx_background_jobs_job_type", table_name="background_jobs")
    op.drop_index("idx_background_jobs_status", table_name="background_jobs")
    op.drop_index("idx_config_parameters_category", table_name="config_parameters")
    op.drop_index("idx_api_keys_user_id", table_name="api_keys")
    op.drop_index("idx_messages_conv", table_name="messages")
    op.drop_index("idx_conversations_user", table_name="conversations")
    op.drop_index("idx_documents_status", table_name="documents")
    op.drop_index("idx_documents_visibility", table_name="documents")
    op.drop_index("idx_documents_group", table_name="documents")
    op.drop_index("idx_documents_owner", table_name="documents")
    op.drop_index(
        "ux_documents_active_slot",
        table_name="documents",
        postgresql_where=sa.text("status IN ('pending', 'processing', 'done')"),
    )
