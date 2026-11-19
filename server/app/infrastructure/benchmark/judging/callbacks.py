"""Typed callbacks for persisting successful judge metrics."""

from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypedDict


AsyncMetricCallback = Callable[[dict[str, Any]], Awaitable[None]]
SaveStage = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


class AsyncMetricOptions(TypedDict, total=False):
    saved_metrics: dict[str, Any] | None
    metric_callback: AsyncMetricCallback | None


class GroupedMetricOptions(TypedDict, total=False):
    saved: dict[str, Any] | None
    callback: AsyncMetricCallback | None
