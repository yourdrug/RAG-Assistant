"""merge heads

Revision ID: 69930d1c204f
Revises: b2c3d4e5f6a8, j5k6l7m8n9o0
Create Date: 2026-08-14 17:56:32.687772

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "69930d1c204f"
down_revision: str | Sequence[str] | None = ("b2c3d4e5f6a8", "j5k6l7m8n9o0")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
