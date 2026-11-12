"""Chunking observations, isolated from clients and pipeline initialization."""

from prometheus_client import Counter


CHUNK_FALLBACK_SPLIT_TOTAL = Counter(
    "chunk_fallback_split_total", "Oversized sentences requiring word or character splitting", ["domain"]
)
BOUNDARY_CANDIDATES_REJECTED_TOTAL = Counter(
    "boundary_candidates_rejected_total", "Rejected structural boundary candidates", ["domain", "rule"]
)
BOUNDARY_NUMBER_GAPS_TOTAL = Counter(
    "boundary_number_gaps_total", "Accepted gaps of one structural number", ["domain"]
)
