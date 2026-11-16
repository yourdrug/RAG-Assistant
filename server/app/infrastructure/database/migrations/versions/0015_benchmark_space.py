"""Persist the knowledge space selected for a benchmark sweep."""

from alembic import op
import sqlalchemy as sa


revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_sweeps", sa.Column("space_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_benchmark_sweeps_space_id",
        "benchmark_sweeps",
        "knowledge_spaces",
        ["space_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    # Freeze old sweeps only if their former implicit space is unambiguous.
    op.execute(
        "UPDATE benchmark_sweeps SET space_id = (SELECT min(id) FROM knowledge_spaces) "
        "WHERE (SELECT count(*) FROM knowledge_spaces) = 1"
    )


def downgrade() -> None:
    op.drop_constraint("fk_benchmark_sweeps_space_id", "benchmark_sweeps", type_="foreignkey")
    op.drop_column("benchmark_sweeps", "space_id")
