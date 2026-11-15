"""Replace TEI/DeepInfra provider settings with generic embedding/rerank endpoints.

The provider was chosen by the ``ML_PROVIDER`` env, so endpoint URLs cannot be derived
from the database: new keys are seeded with defaults at bootstrap. Only the reranker
model choice is carried over.
"""

from alembic import op
import sqlalchemy as sa


revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

LEGACY_KEYS = (
    "ml_provider",
    "tei_embed_url",
    "tei_rerank_url",
    "deepinfra_base_url",
    "deepinfra_embed_model",
    "deepinfra_rerank_model",
)
NEW_KEYS = ("embedding_base_url", "rerank_base_url", "rerank_model", "rerank_api_format")


def delete_keys(keys: tuple[str, ...]) -> None:
    op.execute(
        sa.text("DELETE FROM config_parameters WHERE key IN :keys").bindparams(
            sa.bindparam("keys", keys, expanding=True)
        )
    )


def upgrade() -> None:
    op.execute(
        sa.text(
            "INSERT INTO config_parameters (key, value, value_type, category) "
            "SELECT 'rerank_model', value, 'str', 'ml' FROM config_parameters "
            "WHERE key = 'deepinfra_rerank_model' AND domain_key IS NULL "
            "ON CONFLICT ON CONSTRAINT ux_config_parameters_key_domain DO NOTHING"
        )
    )
    delete_keys(LEGACY_KEYS)


def downgrade() -> None:
    # The previous release reseeds its provider defaults on bootstrap.
    delete_keys(NEW_KEYS)
