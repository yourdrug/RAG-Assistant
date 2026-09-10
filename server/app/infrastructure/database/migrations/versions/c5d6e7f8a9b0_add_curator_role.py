"""Add curator role + scope assignment tables.

- Extend UserRole: admin | curator | user
- Replace chk_client_not_admin with chk_client_not_privileged
- Create curator_user_assignments and curator_group_assignments tables

Revision ID: c5d6e7f8a9b0
Revises: z8d9e0f1g2h3
Create Date: 2026-09-10
"""

from alembic import op
import sqlalchemy as sa


revision = "c5d6e7f8a9b0"
down_revision = "z8d9e0f1g2h3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Drop old constraints
    op.drop_constraint("chk_client_not_admin", "users", type_="check")
    op.drop_constraint("users_role_check", "users", type_="check")

    # 2. Add new constraints
    op.create_check_constraint(
        "users_role_check",
        "users",
        "role IN ('admin', 'curator', 'user')",
    )
    op.create_check_constraint(
        "chk_client_not_privileged",
        "users",
        "NOT (kind = 'client' AND role IN ('admin', 'curator'))",
    )

    # 3. Create curator_user_assignments
    op.create_table(
        "curator_user_assignments",
        sa.Column(
            "curator_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column(
            "target_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("assigned_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column(
            "assigned_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.CheckConstraint("curator_id != target_user_id", name="chk_curator_not_self_assign"),
    )
    op.create_index(
        "idx_curator_user_assignments_target",
        "curator_user_assignments",
        ["target_user_id"],
    )

    # 4. Create curator_group_assignments
    op.create_table(
        "curator_group_assignments",
        sa.Column(
            "curator_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("group_id", sa.Integer(), sa.ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("assigned_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column(
            "assigned_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("curator_group_assignments")
    op.drop_index("idx_curator_user_assignments_target", table_name="curator_user_assignments")
    op.drop_table("curator_user_assignments")
    op.drop_constraint("chk_client_not_privileged", "users", type_="check")
    op.drop_constraint("users_role_check", "users", type_="check")
    op.create_check_constraint(
        "users_role_check",
        "users",
        "role IN ('admin', 'user')",
    )
    op.create_check_constraint(
        "chk_client_not_admin",
        "users",
        "NOT (kind = 'client' AND role = 'admin')",
    )
