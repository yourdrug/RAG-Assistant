"""Fix category for feature toggle config parameters.

These params were originally seeded with category='rag' but should be
in category='toggles'. The startup seeding only checks key existence
and never corrects category, so this must be done via migration.

Revision ID: w5x6y7z8a9b0
Revises: v4w5x6y7z8a9
Create Date: 2026-09-04
"""

from alembic import op

revision = "w5x6y7z8a9b0"
down_revision = "v4w5x6y7z8a9"
branch_labels = None
depends_on = None

TOGGLE_KEYS = (
    "relevance_gate_enabled",
    "condense_enabled",
    "decomposition_enabled",
    "rolling_summary_enabled",
    "cache_enabled",
)


def upgrade() -> None:
    placeholders = ", ".join(f"'{k}'" for k in TOGGLE_KEYS)
    op.execute(
        f"UPDATE config_parameters SET category = 'toggles' "
        f"WHERE key IN ({placeholders}) AND category != 'toggles'"
    )


def downgrade() -> None:
    placeholders = ", ".join(f"'{k}'" for k in TOGGLE_KEYS)
    op.execute(
        f"UPDATE config_parameters SET category = 'rag' "
        f"WHERE key IN ({placeholders})"
    )
