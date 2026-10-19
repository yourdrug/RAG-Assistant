"""Stateless BM25 encoding; Qdrant maintains postings and live IDF statistics."""

from collections import Counter
from hashlib import blake2s

from qdrant_client.models import NamedSparseVector, SparseVector

from infrastructure.bm25.tokenizer import tokenize


SPARSE_VECTOR_NAME = "bm25_ru_v1"
BM25_K1 = 1.5
BM25_B = 0.75
# Fixed normalization avoids rewriting all vectors when corpus size changes.
BM25_AVERAGE_LENGTH = 256.0


def term_id(token: str) -> int:
    """Stable 32-bit vocabulary shared by all processes and corpus generations."""
    return int.from_bytes(blake2s(token.encode(), digest_size=4).digest(), "little")


def encode_sparse(text: str, *, query: bool = False) -> SparseVector:
    counts = Counter(tokenize(text))
    length = sum(counts.values())
    weights: dict[int, float] = {}
    normalization = BM25_K1 * (1 - BM25_B + BM25_B * length / BM25_AVERAGE_LENGTH)

    for token, frequency in counts.items():
        index = term_id(token)
        weight = float(frequency) if query else frequency * (BM25_K1 + 1) / (frequency + normalization)
        # Hash collisions share a dimension, as with other hashed sparse encoders.
        weights[index] = weights.get(index, 0.0) + weight

    indices = sorted(weights)
    return SparseVector(indices=indices, values=[weights[index] for index in indices])


def sparse_query(text: str) -> NamedSparseVector:
    return NamedSparseVector(name=SPARSE_VECTOR_NAME, vector=encode_sparse(text, query=True))


def point_vectors(text: str, dense: list[float]) -> dict[str, list[float] | SparseVector | list[list[float]]]:
    return {"": dense, SPARSE_VECTOR_NAME: encode_sparse(text)}
