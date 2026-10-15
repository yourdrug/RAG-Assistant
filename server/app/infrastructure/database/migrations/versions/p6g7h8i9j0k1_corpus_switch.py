"""Minimal durable corpus switch journal and per-process readiness."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = 'p6g7h8i9j0k1'
down_revision = 'o5f6g7h8i9j0'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'corpus_control',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('phase', sa.Text(), nullable=False),
        sa.Column('write_epoch', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('active', JSONB(), nullable=False),
        sa.Column('candidate', JSONB()),
        sa.Column('digest', sa.Text()),
        sa.Column('owner', sa.Text()),
        sa.Column('error', sa.Text()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint('id = 1'),
        sa.CheckConstraint("phase IN ('active', 'preparing', 'committing', 'failed')"),
    )
    op.create_table(
        'corpus_processes',
        sa.Column('process_id', sa.Text(), primary_key=True),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('prepared_generation', sa.Text()),
        sa.Column('applied_generation', sa.Text()),
        sa.Column('error', sa.Text()),
        sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("role IN ('api', 'worker', 'cli')"),
    )


def downgrade():
    op.drop_table('corpus_processes')
    op.drop_table('corpus_control')
