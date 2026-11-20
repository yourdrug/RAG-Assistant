"""Remove global runtime parameters for the retired local OCR engines."""

from alembic import op


revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM config_parameters
        WHERE domain_key IS NULL
          AND key IN ('ocr_engine', 'ocr_lang_paddle', 'ocr_lang_surya', 'ocr_min_confidence')
        """
    )


def downgrade() -> None:
    """The previous release can seed these parameters from its defaults."""
