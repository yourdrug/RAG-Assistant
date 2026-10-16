"""Public policy, admission arguments and budget administration results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict


if TYPE_CHECKING:
    from decimal import Decimal


@dataclass(frozen=True)
class BudgetPolicy:
    enabled: bool = False
    hourly_usd: str = "1.0"
    daily_usd: str = "5.0"


class BudgetOperation(TypedDict):
    policy: BudgetPolicy
    principal: str | None
    operation_id: str
    provider: str
    model: str
    operation: str
    maximum_units: int | None
    rate_version: int


class BudgetReservation(TypedDict):
    principal: str
    operation_id: str
    provider: str
    model: str
    operation: str
    maximum_units: int
    rate_version: int
    hourly_units: int
    daily_units: int
    enforce_limits: NotRequired[bool]
    request_fingerprint: NotRequired[str | None]


class ModelBudgetStatus(TypedDict):
    provider: str
    model: str
    ready: bool
    reason: str | None


class CostRateRow(TypedDict):
    provider: str
    model: str
    version: int
    unit: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    context_tokens: int


BudgetUsageRows = Sequence[Mapping[str, object]]
