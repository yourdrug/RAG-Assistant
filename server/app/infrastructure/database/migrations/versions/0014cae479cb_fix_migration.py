"""fix migration — drop client_assignments and drop orphaned chunks indexes

Revision ID: 0014cae479cb
Revises: l2m3n4o5p6q7
Create Date: 2026-08-23 14:20:09.506939

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0014cae479cb'
down_revision: Union[str, Sequence[str], None] = 'l2m3n4o5p6q7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(conn, table_name: str) -> bool:
    result = conn.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :name)"),
        {"name": table_name},
    ).scalar()
    return result


def _index_exists(conn, index_name: str) -> bool:
    result = conn.execute(
        sa.text("SELECT EXISTS (  SELECT 1 FROM pg_indexes WHERE indexname = :idx)"),
        {"idx": index_name},
    ).scalar()
    return result


def upgrade() -> None:
    """Drop client_assignments table and chunks indexes.

    These are no longer needed after schema restructuring.
    """
    conn = op.get_bind()

    if _table_exists(conn, "client_assignments"):
        op.drop_table("client_assignments")

    for idx_name in [
        "ix_chunks_content_trgm",
        "ix_chunks_group_id",
        "ix_chunks_owner_id",
        "ix_chunks_visibility",
    ]:
        if _index_exists(conn, idx_name):
            op.execute(sa.text(f"DROP INDEX IF EXISTS {idx_name}"))


def downgrade() -> None:
    """Recreate client_assignments and chunks indexes."""
    op.create_index(
        op.f("ix_chunks_visibility"),
        "chunks",
        ["visibility"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chunks_owner_id"),
        "chunks",
        ["owner_id"],
        unique=False,
        postgresql_where="(owner_id IS NOT NULL)",
    )
    op.create_index(
        op.f("ix_chunks_group_id"),
        "chunks",
        ["group_id"],
        unique=False,
        postgresql_where="(group_id IS NOT NULL)",
    )
    op.execute("CREATE INDEX ix_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops)")

    op.create_table(
        "client_assignments",
        sa.Column("internal_user_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("client_user_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("assigned_by", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("assigned_at", postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
        sa.ForeignKeyConstraint(
            ["assigned_by"],
            ["users.id"],
            name=op.f("client_assignments_assigned_by_fkey"),
        ),
        sa.ForeignKeyConstraint(
            ["client_user_id"],
            ["users.id"],
            name=op.f("client_assignments_client_user_id_fkey"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["internal_user_id"],
            ["users.id"],
            name=op.f("client_assignments_internal_user_id_fkey"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "internal_user_id",
            "client_user_id",
            name=op.f("client_assignments_pkey"),
        ),
    )
