"""Answer style and LLM finish reason of each chat attempt.

``answer_style`` is the presentation the user asked for (concise/adaptive/detailed) and
``finish_reason`` the provider's stop reason, so truncated answers can be found and compared
across styles. Both are nullable: older rows and attempts without generation have no value.
"""

from alembic import op
import sqlalchemy as sa


revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_logs", sa.Column("answer_style", sa.String(20), nullable=True))
    op.add_column("chat_logs", sa.Column("finish_reason", sa.String(30), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_logs", "finish_reason")
    op.drop_column("chat_logs", "answer_style")
