"""merge heads: curator role + document versioning

Revision ID: 08a6c8d44aea
Revises: c1d2e3f4a5b6, c5d6e7f8a9b0
Create Date: 2026-09-10 11:26:16.538389

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = '08a6c8d44aea'
down_revision: Union[str, Sequence[str], None] = ('c1d2e3f4a5b6', 'c5d6e7f8a9b0')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
