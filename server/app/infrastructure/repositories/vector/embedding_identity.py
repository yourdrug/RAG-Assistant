"""Persist an immutable model binding beside each physical Qdrant collection."""

from __future__ import annotations

import hashlib

from qdrant_client.models import Distance, VectorParams

from config import settings


def configured_embedding_identity() -> str:
    model = (
        settings.deepinfra_embed_model if settings.ml_provider == "deepinfra" else settings.tei_embed_model
    )
    return f"{settings.ml_provider}:{model}:{settings.embedding_revision}"


def ensure_embedding_identity(client, identity: str | None = None) -> str:
    """Reject model changes and unversioned nonempty corpora before reads/writes.

    A separate empty collection stores the fingerprint as a named vector config.
    Collection creation is atomic, including competing worker startups. Existing
    bindings are never overwritten; migration must create a fresh corpus.
    """
    identity = identity or configured_embedding_identity()
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    corpus = settings.collection_name
    names = {c.name for c in client.get_collections().collections}
    if corpus not in names:
        aliases = client.get_aliases().aliases
        corpus = next((a.collection_name for a in aliases if a.alias_name == corpus), corpus)
    binding = f"{corpus}__embedding_identity"
    if binding not in names:
        if corpus in names and client.get_collection(corpus).points_count:
            raise ValueError(
                f"Collection {corpus!r} contains unversioned embeddings. "
                "Re-index into a new collection before enabling model identity validation."
            )
        try:
            client.create_collection(
                collection_name=binding,
                vectors_config={fingerprint: VectorParams(size=1, distance=Distance.COSINE)},
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
    vectors = client.get_collection(binding).config.params.vectors
    if not isinstance(vectors, dict) or set(vectors) != {fingerprint}:
        raise ValueError(
            f"Collection {corpus!r} belongs to a different embedding model. "
            "Re-index into a new collection and switch the alias after validation."
        )
    return identity
