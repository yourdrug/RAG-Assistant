"""Drop USD cost budgeting: tariffs, usage ledger and its runtime settings."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

COST_SETTING_KEYS = ("cost_rate_limit_enabled", "cost_hourly_limit", "cost_daily_limit")


def upgrade() -> None:
    op.execute(
        sa.text("DELETE FROM config_parameters WHERE key IN :keys").bindparams(
            sa.bindparam("keys", COST_SETTING_KEYS, expanding=True)
        )
    )
    op.drop_index("ix_cost_usage_principal_window", table_name="cost_usage", if_exists=True)
    op.drop_table("cost_usage", if_exists=True)
    op.drop_table("cost_rates", if_exists=True)


def downgrade() -> None:
    # Restores the empty schema; the previous release reseeds its default settings on bootstrap.
    op.create_table(
        "cost_rates",
        sa.Column("provider", sa.TEXT(), nullable=False),
        sa.Column("model", sa.TEXT(), nullable=False),
        sa.Column("version", sa.INTEGER(), nullable=False),
        sa.Column("unit", sa.TEXT(), nullable=False),
        sa.Column("input_usd_per_million", sa.NUMERIC(precision=24, scale=12), nullable=False),
        sa.Column("output_usd_per_million", sa.NUMERIC(precision=24, scale=12), nullable=False),
        sa.Column("context_tokens", sa.INTEGER(), nullable=False),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.CheckConstraint(
            "input_usd_per_million >= 0::numeric AND output_usd_per_million >= 0::numeric",
            name="cost_rates_check1",
        ),
        sa.CheckConstraint("version > 0 AND context_tokens > 0", name="cost_rates_check"),
        sa.PrimaryKeyConstraint("provider", "model", "version", name="cost_rates_pkey"),
    )
    op.create_table(
        "cost_usage",
        sa.Column("operation_id", sa.TEXT(), nullable=False),
        sa.Column("principal", sa.TEXT(), nullable=False),
        sa.Column("provider", sa.TEXT(), nullable=False),
        sa.Column("model", sa.TEXT(), nullable=False),
        sa.Column("operation", sa.TEXT(), nullable=False),
        sa.Column("provider_response_id", sa.TEXT(), nullable=True),
        sa.Column("reconciliation_evidence", sa.TEXT(), nullable=True),
        sa.Column("rate_version", sa.INTEGER(), nullable=False),
        sa.Column("reserved_units", sa.BIGINT(), nullable=False),
        sa.Column("actual_units", sa.BIGINT(), nullable=True),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("settled_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("request_fingerprint", sa.TEXT(), nullable=True),
        sa.Column("billable_units", postgresql.JSONB, nullable=True),
        sa.Column("cost_source", sa.TEXT(), nullable=True),
        sa.Column("dispatch_state", sa.TEXT(), server_default=sa.text("'reserved'::text"), nullable=False),
        sa.CheckConstraint(
            "dispatch_state = ANY (ARRAY['reserved'::text, 'sending'::text, 'settled'::text])",
            name="ck_cost_usage_dispatch",
        ),
        sa.CheckConstraint("(actual_units IS NULL) = (settled_at IS NULL)", name="cost_usage_check1"),
        sa.CheckConstraint(
            "reserved_units >= 0 AND (actual_units IS NULL OR actual_units >= 0)", name="cost_usage_check"
        ),
        sa.ForeignKeyConstraint(
            ["provider", "model", "rate_version"],
            ["cost_rates.provider", "cost_rates.model", "cost_rates.version"],
            name="cost_usage_provider_model_rate_version_fkey",
        ),
        sa.PrimaryKeyConstraint("operation_id", name="cost_usage_pkey"),
    )
    op.create_index("ix_cost_usage_principal_window", "cost_usage", ["principal", "settled_at"], unique=False)
