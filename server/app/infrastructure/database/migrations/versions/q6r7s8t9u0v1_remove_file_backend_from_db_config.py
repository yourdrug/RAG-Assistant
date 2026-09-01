"""Remove file_backend and data_dir from config_parameters

Both are static parameters managed via .env only. Changing them requires
a server restart, so they should not be in the dynamic config table.

Revision ID: q6r7s8t9u0v1
Revises: p5q6r7s8t9u0
Create Date: 2026-08-29
"""

from alembic import op

revision = "q6r7s8t9u0v1"
down_revision = "p5q6r7s8t9u0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM config_parameters WHERE key IN ('file_backend', 'data_dir')")


def downgrade() -> None:
    # Default values are seeded at startup from settings (initialization.py)
    pass
