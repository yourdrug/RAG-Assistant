"""Remove the retired ``retriever_top_k`` runtime parameter.

Chat and benchmarks read only ``retriever_top_k_broad``; a saved narrow value is ignored by the
configuration loader and would only show up as a stale row. Downgrade cannot restore it: the old
code falls back to its default.
"""

from alembic import op


revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM config_parameters WHERE key = 'retriever_top_k'")


def downgrade() -> None:
    """The removed value is not recoverable; the previous release uses its code default."""
