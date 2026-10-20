"""Benchmark sweep lifecycle status."""

from __future__ import annotations

from enum import StrEnum

from domain.shared.enums import LifecycleStatusMixin


class BenchmarkSweepStatus(
    LifecycleStatusMixin,
    StrEnum,
    terminal=frozenset({"done", "failed", "cancelled"}),
    active=frozenset({"pending", "running"}),
):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
