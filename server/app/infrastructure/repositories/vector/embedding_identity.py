"""Persist an immutable model binding beside each physical Qdrant collection."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from qdrant_client.models import Distance, VectorParams

from config import settings
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.ml.config.embedding_identity import configured_embedding_identity


if TYPE_CHECKING:
    from infrastructure.repositories.vector.client_types import VectorClient


# Bindings are immutable, so a pinned snapshot verified once stays valid; keyed by
# snapshot namespace (endpoint, physical collection, generation) and fingerprint.
verified_snapshot_bindings: set[tuple[str, str]] = set()


def ensure_embedding_identity(client: VectorClient, identity: str | None = None) -> str:
    """Reject model changes and unversioned nonempty corpora before reads/writes.

    A separate empty collection stores the fingerprint as a named vector config.
    Collection creation is atomic, including competing worker startups. Existing
    bindings are never overwritten; migration must create a fresh corpus.
    """
    snapshot = getattr(client, "snapshot", None)
    identity = identity or (
        snapshot.embedding_identity
        if isinstance(snapshot, CorpusSnapshot)
        else configured_embedding_identity()
    )
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    cache_key = (snapshot.namespace, fingerprint) if isinstance(snapshot, CorpusSnapshot) else None

    if cache_key is not None and cache_key in verified_snapshot_bindings:
        return identity

    verify_binding(client, snapshot, fingerprint)

    if cache_key is not None:
        verified_snapshot_bindings.add(cache_key)

    return identity


def verify_binding(client: VectorClient, snapshot: object, fingerprint: str) -> None:
    corpus = (
        snapshot.physical_collection if isinstance(snapshot, CorpusSnapshot) else settings.collection_name
    )
    names = {c.name for c in client.get_collections().collections}

    if corpus not in names:
        aliases = client.get_aliases().aliases
        corpus = next((a.collection_name for a in aliases if a.alias_name == corpus), corpus)

    binding = f"{corpus}__embedding_identity"

    if binding not in names:
        if corpus in names and client.get_collection(corpus).points_count:
            msg = (
                f"Collection {corpus!r} contains unversioned embeddings. "
                "Re-index into a new collection before enabling model identity validation."
            )
            raise ValueError(msg)

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
        msg = (
            f"Collection {corpus!r} belongs to a different embedding model. "
            "Re-index into a new collection and switch the alias after validation."
        )
        raise ValueError(msg)
