"""Shared constants for the presentation layer.

Centralises magic strings, numeric limits, and reusable configuration
that were previously scattered across route handlers, schemas, and
dependency providers.
"""

from __future__ import annotations

from enum import StrEnum


# ---------------------------------------------------------------------------
# Auth scheme names
# ---------------------------------------------------------------------------
AUTH_SCHEME_BEARER = "bearer"
AUTH_SCHEME_API_KEY = "api-key"


# ---------------------------------------------------------------------------
# Job types (must match worker task names)
# ---------------------------------------------------------------------------
class JobType(StrEnum):
    DOCUMENT_PROCESSING = "document_processing"
    INGEST = "ingest"
    BENCHMARK = "benchmark"
    SWEEP = "sweep"


# ---------------------------------------------------------------------------
# Pagination defaults and limits
# ---------------------------------------------------------------------------
DEFAULT_PAGE_LIMIT = 50
MIN_PAGE_LIMIT = 1
MAX_PAGE_LIMIT = 200
MAX_PAGE_LIMIT_LARGE = 500
DEFAULT_PAGE_OFFSET = 0


# ---------------------------------------------------------------------------
# Upload / file size
# ---------------------------------------------------------------------------
FILE_TOO_LARGE_STATUS = 413


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------
SSE_MEDIA_TYPE = "text/event-stream"
SSE_HEADERS: dict[str, str] = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
SSE_HEARTBEAT = ": heartbeat\n\n"


# ---------------------------------------------------------------------------
# Question / content truncation
# ---------------------------------------------------------------------------
QUESTION_LOG_MAX_CHARS = 100
PAGE_PREVIEW_MAX_CHARS = 200
FULL_TEXT_PREVIEW_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
PAGE_IMAGE_DPI = 120


# ---------------------------------------------------------------------------
# Confidence metadata key (filtered from sources in responses)
# ---------------------------------------------------------------------------
CONFIDENCE_KEY = "_confidence"


# ---------------------------------------------------------------------------
# Magic byte signatures for file validation
# ---------------------------------------------------------------------------
MAGIC_BYTES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04"],  # ZIP-based
    ".doc": [b"\xd0\xcf\x11\xe0"],  # OLE2
    ".rtf": [b"{\\rtf"],
    ".md": [],  # plain text, no reliable magic
    ".txt": [],  # plain text
}


# ---------------------------------------------------------------------------
# Config keys that are static (not editable via API)
# ---------------------------------------------------------------------------
STATIC_CONFIG_KEYS: frozenset[str] = frozenset({"file_backend", "data_dir"})


# ---------------------------------------------------------------------------
# Benchmark run config → live config parameter mapping
# (canonical location: application.services.benchmark_services)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Default quality warning threshold (bad-page ratio)
# ---------------------------------------------------------------------------
QUALITY_BAD_RATIO_THRESHOLD: float = 0.3
