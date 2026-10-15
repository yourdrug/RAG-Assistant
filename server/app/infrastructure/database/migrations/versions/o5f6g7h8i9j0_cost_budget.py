"""Versioned USD tariffs and durable pending/settled usage (no seeded prices)."""

from alembic import op
import sqlalchemy as sa

revision = 'o5f6g7h8i9j0'
down_revision = 'n4e5f6g7h8i9'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'cost_rates',
        sa.Column('provider', sa.Text(), primary_key=True),
        sa.Column('model', sa.Text(), primary_key=True),
        sa.Column('version', sa.Integer(), primary_key=True),
        sa.Column('unit', sa.Text(), nullable=False),
        sa.Column('input_usd_per_million', sa.Numeric(24, 12), nullable=False),
        sa.Column('output_usd_per_million', sa.Numeric(24, 12), nullable=False),
        sa.Column('context_tokens', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('version > 0 AND context_tokens > 0'),
        sa.CheckConstraint('input_usd_per_million >= 0 AND output_usd_per_million >= 0'),
    )
    op.create_table(
        'cost_usage',
        sa.Column('operation_id', sa.Text(), primary_key=True),
        sa.Column('principal', sa.Text(), nullable=False),
        sa.Column('provider', sa.Text(), nullable=False),
        sa.Column('model', sa.Text(), nullable=False),
        sa.Column('operation', sa.Text(), nullable=False),
        sa.Column('provider_response_id', sa.Text()),
        sa.Column('reconciliation_evidence', sa.Text()),
        sa.Column('rate_version', sa.Integer(), nullable=False),
        sa.Column('reserved_units', sa.BigInteger(), nullable=False),
        sa.Column('actual_units', sa.BigInteger()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('settled_at', sa.DateTime(timezone=True)),
        sa.CheckConstraint('reserved_units >= 0 AND (actual_units IS NULL OR actual_units >= 0)'),
        sa.CheckConstraint('(actual_units IS NULL) = (settled_at IS NULL)'),
        sa.ForeignKeyConstraint(
            ['provider', 'model', 'rate_version'],
            ['cost_rates.provider', 'cost_rates.model', 'cost_rates.version'],
        ),
    )
    op.create_index('ix_cost_usage_principal_window', 'cost_usage', ['principal', 'settled_at'])


def downgrade():
    op.drop_table('cost_usage')
    op.drop_table('cost_rates')
