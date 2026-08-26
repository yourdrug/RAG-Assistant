"""Add token usage to chat_logs

Revision ID: n3o4p5q6r7s8
Revises: m1n2o3p4q5r6
Create Date: 2026-08-26
"""

from alembic import op
import sqlalchemy as sa


revision = "n3o4p5q6r7s8"
down_revision = "m1n2o3p4q5r6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_logs", sa.Column("input_tokens", sa.Integer(), nullable=True))
    op.add_column("chat_logs", sa.Column("output_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_logs", "output_tokens")
    op.drop_column("chat_logs", "input_tokens")
