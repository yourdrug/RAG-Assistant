"""Shared queue event counters survive worker restarts and job-row cleanup."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
from time import perf_counter
from typing import TYPE_CHECKING, cast

from prometheus_client import REGISTRY, Histogram
from prometheus_client.core import CounterMetricFamily

from domain.runtime.jobs import JobType
from infrastructure.redis.redis_client import redis_client


if TYPE_CHECKING:
    from collections.abc import Awaitable


logger = logging.getLogger(__name__)
EVENTS = {
    "deferred": "background_job_limit_deferrals",
    "lease_expired": "background_job_lease_expirations",
    "republished": "background_job_republications",
    "cleaned": "background_job_cleanup_rows",
}
EVENT_TYPES = {event: ("all",) if event == "cleaned" else tuple(JobType) for event in EVENTS}
JOB_FENCING = Histogram("background_job_fencing_seconds", "Duration of execution-owner SQL checks")


@dataclass
class FencingMeasurements:
    checks: int = 0
    seconds: float = 0


fencing_measurements: ContextVar[FencingMeasurements | None] = ContextVar("job_fencing", default=None)


@contextmanager
def measure_fencing() -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - started
        JOB_FENCING.observe(duration)
        measurements = fencing_measurements.get()

        if measurements is not None:
            measurements.checks += 1
            measurements.seconds += duration


class JobEventCollector:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, float]] = {
            event: dict.fromkeys(EVENT_TYPES[event], 0.0) for event in EVENTS
        }

    def collect(self) -> Iterator[CounterMetricFamily]:
        for event, name in EVENTS.items():
            metric = CounterMetricFamily(name, f"Shared background job events: {event}", labels=["type"])

            for job_type, value in self.values.get(event, {}).items():
                metric.add_metric([job_type], value)

            yield metric


job_events = JobEventCollector()
REGISTRY.register(job_events)


async def record_job_event(event: str, job_type: str, count: int = 1) -> None:
    try:
        async with asyncio.timeout(1):
            await cast(
                "Awaitable[int]", redis_client.async_redis.hincrby(f"rag:job_events:{event}", job_type, count)
            )
    except Exception as exc:  # noqa: BLE001 -- telemetry must not change job outcomes
        logger.warning("Job telemetry unavailable: event=%s error_type=%s", event, type(exc).__name__)


async def collect_job_events() -> None:
    async with asyncio.timeout(3):
        for event in EVENTS:
            values = await cast(
                "Awaitable[dict[str, str]]", redis_client.async_redis.hgetall(f"rag:job_events:{event}")
            )
            job_events.values[event] = {
                **dict.fromkeys(EVENT_TYPES[event], 0.0),
                **{str(key): float(value) for key, value in values.items()},
            }
