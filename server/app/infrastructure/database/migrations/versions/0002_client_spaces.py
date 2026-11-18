"""A client may be assigned several support spaces instead of one.

Downgrade keeps one assigned space per client: the one with the smallest id.
"""

from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "client_spaces",
        sa.Column("user_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("space_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="client_spaces_user_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["space_id"], ["knowledge_spaces.id"], name="client_spaces_space_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "space_id", name="client_spaces_pkey"),
    )
    op.create_index("idx_client_spaces_space", "client_spaces", ["space_id"], unique=False)
    op.execute(
        "INSERT INTO client_spaces (user_id, space_id) "
        "SELECT id, client_space_id FROM users "
        "WHERE client_space_id IS NOT NULL AND kind = 'client'"
    )
    op.drop_constraint("chk_client_space_kind", "users", type_="check")
    op.drop_constraint("fk_users_client_space_id", "users", type_="foreignkey")
    op.drop_column("users", "client_space_id")


def downgrade() -> None:
    op.add_column("users", sa.Column("client_space_id", sa.INTEGER(), autoincrement=False, nullable=True))
    op.create_foreign_key(
        "fk_users_client_space_id",
        "users",
        "knowledge_spaces",
        ["client_space_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "chk_client_space_kind", "users", "client_space_id IS NULL OR kind::text = 'client'::text"
    )
    op.execute(
        "UPDATE users SET client_space_id = assigned.space_id "
        "FROM (SELECT user_id, MIN(space_id) AS space_id FROM client_spaces GROUP BY user_id) AS assigned "
        "WHERE users.id = assigned.user_id"
    )
    op.drop_index("idx_client_spaces_space", table_name="client_spaces")
    op.drop_table("client_spaces")
