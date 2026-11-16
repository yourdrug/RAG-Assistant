"""Unify LLM provider settings into one OpenAI-compatible endpoint.

The active provider's endpoint and models move to ``llm_base_url``, ``llm_model`` and
``llm_auxiliary_model``; provider-specific keys are removed. Missing values are left
to the bootstrap seeding defaults.
"""

from alembic import op
import sqlalchemy as sa


revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

CATEGORY = "llm_connection"
NEW_KEYS = ("llm_base_url", "llm_model", "llm_auxiliary_model")
LEGACY_KEYS = (
    "llm_provider",
    "openrouter_base_url",
    "openrouter_model",
    "openai_compatible_base_url",
    "openai_compatible_model",
    "openai_compatible_auxiliary_model",
)
OPENAI_COMPATIBLE = "openai_compatible"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def global_values(keys: tuple[str, ...]) -> dict[str, str]:
    rows = op.get_bind().execute(
        sa.text(
            "SELECT key, value FROM config_parameters WHERE domain_key IS NULL AND key IN :keys"
        ).bindparams(sa.bindparam("keys", keys, expanding=True))
    )
    return {row.key: row.value for row in rows}


def insert_missing(values: dict[str, str], category: str) -> None:
    for key, value in values.items():
        op.execute(
            sa.text(
                "INSERT INTO config_parameters (key, value, value_type, category) "
                "VALUES (:key, :value, 'str', :category) "
                "ON CONFLICT ON CONSTRAINT ux_config_parameters_key_domain DO NOTHING"
            ).bindparams(key=key, value=value, category=category)
        )


def delete_keys(keys: tuple[str, ...]) -> None:
    op.execute(
        sa.text("DELETE FROM config_parameters WHERE key IN :keys").bindparams(
            sa.bindparam("keys", keys, expanding=True)
        )
    )


def upgrade() -> None:
    legacy = global_values(LEGACY_KEYS)

    if legacy.get("llm_provider") == OPENAI_COMPATIBLE:
        migrated = {
            "llm_base_url": legacy.get("openai_compatible_base_url"),
            "llm_model": legacy.get("openai_compatible_model"),
            "llm_auxiliary_model": legacy.get("openai_compatible_auxiliary_model"),
        }
    else:
        # OpenRouter used a hard-coded fast auxiliary model; it now reuses the answer model.
        migrated = {
            "llm_base_url": legacy.get("openrouter_base_url"),
            "llm_model": legacy.get("openrouter_model"),
        }

    insert_missing({key: value for key, value in migrated.items() if value is not None}, CATEGORY)
    delete_keys(LEGACY_KEYS)


def downgrade() -> None:
    current = global_values(NEW_KEYS)
    base_url = current.get("llm_base_url", OPENROUTER_BASE_URL)

    if base_url.startswith("https://openrouter.ai/"):
        restored = {"llm_provider": "openrouter", "openrouter_base_url": base_url}

        if "llm_model" in current:
            restored["openrouter_model"] = current["llm_model"]
    else:
        restored = {
            "llm_provider": OPENAI_COMPATIBLE,
            "openai_compatible_base_url": base_url,
            "openai_compatible_model": current.get("llm_model", ""),
            "openai_compatible_auxiliary_model": current.get("llm_auxiliary_model", ""),
        }

    for key, value in restored.items():
        if key == "llm_provider":
            category = "llm"
        elif key.startswith("openrouter_"):
            category = "openrouter"
        else:
            category = OPENAI_COMPATIBLE

        insert_missing({key: value}, category)

    delete_keys(NEW_KEYS)
