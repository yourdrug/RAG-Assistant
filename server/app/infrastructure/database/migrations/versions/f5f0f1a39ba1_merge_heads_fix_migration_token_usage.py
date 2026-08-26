"""merge heads: fix migration + token usage

Revision ID: f5f0f1a39ba1
Revises: 0014cae479cb, n3o4p5q6r7s8
Create Date: 2026-08-26 12:35:06.351643

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = 'f5f0f1a39ba1'
down_revision: Union[str, Sequence[str], None] = ('0014cae479cb', 'n3o4p5q6r7s8')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
