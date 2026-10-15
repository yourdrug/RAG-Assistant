"""Preserve tariffs and usage while adding physical-attempt audit data."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'q7h8i9j0k1l2'
down_revision = 'p6g7h8i9j0k1'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('cost_usage', sa.Column('request_fingerprint', sa.Text()))
    op.add_column('cost_usage', sa.Column('billable_units', postgresql.JSONB()))
    op.add_column('cost_usage', sa.Column('cost_source', sa.Text()))
    # Existing pending rows may already have been sent: never label them unsent.
    op.add_column(
        'cost_usage', sa.Column('dispatch_state', sa.Text(), server_default='sending', nullable=False)
    )
    op.execute("UPDATE cost_usage SET dispatch_state='settled' WHERE settled_at IS NOT NULL")
    op.alter_column('cost_usage', 'dispatch_state', server_default='reserved')
    op.create_check_constraint(
        'ck_cost_usage_dispatch', 'cost_usage', "dispatch_state IN ('reserved', 'sending', 'settled')"
    )


def downgrade():
    op.drop_constraint('ck_cost_usage_dispatch', 'cost_usage')
    for column in ('dispatch_state', 'cost_source', 'billable_units', 'request_fingerprint'):
        op.drop_column('cost_usage', column)
