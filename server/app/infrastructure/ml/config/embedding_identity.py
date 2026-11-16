"""Compute the configured model binding without importing vector implementations."""

from config import EmbeddingIdentitySettings, ml_client_settings


def configured_embedding_identity(model: EmbeddingIdentitySettings | None = None) -> str:
    model = (model or ml_client_settings()).model_copy()
    return f"{model.embedding_model}:{model.embedding_revision}"
