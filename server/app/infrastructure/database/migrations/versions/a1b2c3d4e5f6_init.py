"""init — create all tables using BaseModel (int PK + creation_date) and LinkedBaseModel (M2M).

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-07-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all tables (mirrors init.sql)."""
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=16), server_default="user", nullable=False),
        sa.Column("kind", sa.String(length=16), server_default="internal", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("TRUE"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.CheckConstraint(
            "role IN ('admin', 'user')",
            name="users_role_check",
        ),
        sa.CheckConstraint(
            "kind IN ('internal', 'client')",
            name="users_kind_check",
        ),
        sa.CheckConstraint(
            "NOT (kind = 'client' AND role = 'admin')",
            name="chk_client_not_admin",
        ),
    )

    op.create_table(
        "groups",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="messages_role_check"),
    )

    op.create_index("idx_messages_conv", "messages", ["conversation_id"])
    op.create_index("idx_conversations_user", "conversations", ["user_id"])

    op.create_table(
        "user_groups",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("group_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["group_id"], ["groups.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "group_id"),
    )

    op.create_table(
        "client_assignments",
        sa.Column("internal_user_id", sa.Integer(), nullable=False),
        sa.Column("client_user_id", sa.Integer(), nullable=False),
        sa.Column("assigned_by", sa.Integer(), nullable=True),
        sa.Column("assigned_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("internal_user_id", "client_user_id"),
        sa.ForeignKeyConstraint(["internal_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["client_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["assigned_by"], ["users.id"]),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("source_path", sa.Text(), server_default="", nullable=False),
        sa.Column("visibility", sa.String(length=20), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=True),
        sa.Column("group_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("warning_message", sa.Text(), nullable=True),
        sa.Column("chunks", sa.Integer(), nullable=True),
        sa.Column("chars", sa.Integer(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["group_id"], ["groups.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "visibility IN ('internal_public', 'internal_group', 'internal_private', 'client_private')",
            name="documents_visibility_check",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'done', 'failed')",
            name="documents_status_check",
        ),
    )

    op.create_index("idx_documents_owner", "documents", ["owner_id"])
    op.create_index("idx_documents_group", "documents", ["group_id"])
    op.create_index("idx_documents_visibility", "documents", ["visibility"])
    op.create_index("idx_documents_status", "documents", ["status"])
    op.create_index(
        "ux_documents_active_slot",
        "documents",
        ["owner_id", "filename"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing', 'done')"),
    )

    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("key_prefix", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("key_hash"),
    )

    op.create_index("idx_api_keys_user_id", "api_keys", ["user_id"])


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.drop_index("idx_api_keys_user_id", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index("ux_documents_active_slot", table_name="documents")
    op.drop_index("idx_documents_status", table_name="documents")
    op.drop_index("idx_documents_visibility", table_name="documents")
    op.drop_index("idx_documents_group", table_name="documents")
    op.drop_index("idx_documents_owner", table_name="documents")
    op.drop_table("documents")
    op.drop_table("client_assignments")
    op.drop_table("user_groups")
    op.drop_index("idx_conversations_user", table_name="conversations")
    op.drop_index("idx_messages_conv", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("groups")
    op.drop_table("users")
