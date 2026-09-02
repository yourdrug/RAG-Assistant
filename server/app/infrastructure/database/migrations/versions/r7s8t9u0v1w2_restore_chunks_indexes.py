"""Restore chunks indexes dropped by 0014cae479cb.

Revision ID: r7s8t9u0v1w2
Revises: j7k8l9m0n1o2
Create Date: 2026-09-02 12:00:00.000000
"""

from alembic import op

revision: str = "r7s8t9u0v1w2"
down_revision: str | None = "j7k8l9m0n1o2"


def upgrade() -> None:
    # GIN trigram index for substring search (ILIKE '%query%')
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_content_trgm " "ON chunks USING GIN (content gin_trgm_ops)"
    )
    # ACL filter indexes
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_owner_id ON chunks (owner_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_group_id ON chunks (group_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_visibility ON chunks (visibility)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_trgm")
    op.execute("DROP INDEX IF EXISTS ix_chunks_owner_id")
    op.execute("DROP INDEX IF EXISTS ix_chunks_group_id")
    op.execute("DROP INDEX IF EXISTS ix_chunks_visibility")
