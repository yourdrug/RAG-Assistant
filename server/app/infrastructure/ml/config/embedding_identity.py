"""Compute the configured model binding without importing vector implementations."""

from application.runtime.enums import MLProvider
from config import EmbeddingIdentitySettings, ml_client_settings


def configured_embedding_identity(model: EmbeddingIdentitySettings | None = None) -> str:
    model = (model or ml_client_settings()).model_copy()
    name = (
        model.deepinfra_embed_model
        if model.ml_provider == MLProvider.DEEPINFRA.value
        else model.tei_embed_model
    )
    return f"{model.ml_provider}:{name}:{model.embedding_revision}"
