"""Candidate caches passed to sweep strategies without search-client objects."""

from domain.knowledge.chunk import Chunk


DenseCache = dict[str, list[tuple[str, float, Chunk]]]
SparseCache = dict[str, list[tuple[str, float]]]
CandidateCache = dict[str, Chunk]
RerankerCache = dict[str, dict[str, float]]
