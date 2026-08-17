"""Benchmark sweep lifecycle status."""

from __future__ import annotations

from enum import StrEnum


class BenchmarkSweepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
