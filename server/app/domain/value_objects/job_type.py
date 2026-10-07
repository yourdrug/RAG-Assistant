"""Canonical types of durable background jobs."""

from enum import StrEnum


class JobType(StrEnum):
    DOCUMENT_PROCESSING = "document_processing"
    DOCUMENT_DELETION = "document_deletion"
    INGEST = "ingest"
    BENCHMARK = "benchmark"
    SWEEP = "sweep"
