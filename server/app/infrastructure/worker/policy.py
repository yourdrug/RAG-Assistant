"""Execution policy per job type: bounded retries and cluster-wide concurrency.

Broker deliveries never consume execution attempts.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import random

import httpx
from openai import APIConnectionError, APIStatusError

from config import JobsSection, process_section
from domain.runtime.jobs import ConcurrencyLimit, JobType
from domain.shared.exceptions import AppClientError


DEFAULT_MAX_TRIES = 3
# Ingest resumes per-source checkpoints. Paid benchmarks and user-resumable
# sweeps still require an explicit restart after failure.
# A sync run is never retried as such: the next requested or scheduled run converges.
MAX_TRIES: dict[str, int] = {JobType.BENCHMARK: 1, JobType.SWEEP: 1, JobType.SYNC_RUN: 1}
# Job types sharing a setting share one cluster-wide limit (enforced at claim).
GLOBAL_LIMIT_SETTINGS: dict[str, str] = {
    JobType.INGEST: "ingest_max_concurrent",
    JobType.INGEST_FILE: "ingest_max_concurrent",
    JobType.BENCHMARK: "benchmark_job_max_concurrent",
    JobType.SWEEP: "sweep_max_concurrent",
}


@dataclass(frozen=True)
class RetryPolicy:
    max_tries: int = 3
    base_seconds: float = 5
    cap_seconds: float = 60

    def delay(self, error: Exception, attempt: int) -> float | None:  # noqa: PLR0911
        if isinstance(error, (AppClientError, ValueError, TypeError, FileNotFoundError, PermissionError)):
            return None

        if isinstance(error, (httpx.HTTPStatusError, APIStatusError)):
            if (
                error.response.status_code not in {408, 429}
                and error.response.status_code < httpx.codes.INTERNAL_SERVER_ERROR
            ):
                return None

            retry_after = error.response.headers.get("retry-after")

            if retry_after:
                try:
                    minimum = max(0.0, float(retry_after))
                except ValueError:
                    try:
                        minimum = max(
                            0.0, (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
                        )
                    except (ValueError, TypeError):
                        minimum = 0.0
                return minimum + random.uniform(0, self.base_seconds)  # noqa: S311

        if not isinstance(
            error,
            (
                ConnectionError,
                TimeoutError,
                OSError,
                httpx.TransportError,
                httpx.HTTPStatusError,
                APIConnectionError,
                APIStatusError,
            ),
        ):
            return None

        return self.backoff(attempt)

    def backoff(self, attempt: int) -> float:
        base = min(self.cap_seconds, self.base_seconds * 2 ** (attempt - 1))
        return float(base + random.uniform(0, base * 0.2))  # noqa: S311


def max_tries(job_type: str) -> int:
    return MAX_TRIES.get(job_type, DEFAULT_MAX_TRIES)


def concurrency_limit(job_type: str) -> ConcurrencyLimit | None:
    setting = GLOBAL_LIMIT_SETTINGS.get(job_type)
    settings = process_section(JobsSection)

    if setting is None:
        if job_type == JobType.DOCUMENT_PROCESSING:
            return ConcurrencyLimit(
                frozenset({str(job_type)}), max_per_principal=settings.job_max_per_principal
            )

        return None

    group = frozenset(str(other) for other, name in GLOBAL_LIMIT_SETTINGS.items() if name == setting)
    return ConcurrencyLimit(group, getattr(settings, setting), settings.job_max_per_principal)
