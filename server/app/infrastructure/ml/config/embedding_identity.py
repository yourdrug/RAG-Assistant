"""Compute the configured model binding without importing vector implementations."""

from config import Settings, settings
from domain.runtime.enums import MLProvider


def configured_embedding_identity(model: Settings | None = None) -> str:
    model = (model or settings).model_copy()
    name = (
        model.deepinfra_embed_model
        if model.ml_provider == MLProvider.DEEPINFRA.value
        else model.tei_embed_model
    )
    return f"{model.ml_provider}:{name}:{model.embedding_revision}"
