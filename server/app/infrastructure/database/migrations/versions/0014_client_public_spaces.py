"""Common client articles and current client support-space routing."""

from alembic import op
import sqlalchemy as sa


revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("client_space_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_users_client_space_id",
        "users",
        "knowledge_spaces",
        ["client_space_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint("chk_client_space_kind", "users", "client_space_id IS NULL OR kind = 'client'")
    op.drop_constraint("documents_visibility_check", "documents", type_="check")
    op.create_check_constraint(
        "documents_visibility_check",
        "documents",
        "visibility IN ('internal_public', 'internal_group', 'internal_private', "
        "'client_private', 'client_public')",
    )


def downgrade() -> None:
    # The old constraint deliberately refuses downgrade while common client articles exist.
    op.drop_constraint("documents_visibility_check", "documents", type_="check")
    op.create_check_constraint(
        "documents_visibility_check",
        "documents",
        "visibility IN ('internal_public', 'internal_group', 'internal_private', 'client_private')",
    )
    op.drop_constraint("chk_client_space_kind", "users", type_="check")
    op.drop_constraint("fk_users_client_space_id", "users", type_="foreignkey")
    op.drop_column("users", "client_space_id")
