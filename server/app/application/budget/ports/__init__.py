"""Durable admission contract for each physical paid provider attempt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Unpack


if TYPE_CHECKING:
    from application.budget.dto import BudgetReservation, BudgetUsageRows, CostRateRow
    from domain.runtime.cost import CostRate, ModelBillingContract


class CostBudgetPort(Protocol):
    async def reserve(self, **request: Unpack[BudgetReservation]) -> bool:
        """Reserve once; rearm only the same request proven never dispatched.

        False means a previous attempt exists that cannot safely be resent.
        """
        ...

    async def remember_response(self, principal: str, operation_id: str, response_id: str) -> None: ...

    async def mark_dispatch(self, principal: str, operation_id: str) -> None: ...

    async def reconcile(
        self,
        principal: str,
        operation_id: str,
        actual_units: int,
        *,
        billable_units: dict[str, object] | None = None,
        cost_source: str | None = None,
    ) -> None:
        """Idempotently settle usage; unknown outcomes remain reserved."""
        ...


class BudgetAdminRepositoryPort(Protocol):
    async def rates(self) -> list[CostRateRow]: ...
    async def rate_violated(self, provider: str, model: str, version: int) -> bool: ...
    async def add_rate(self, rate: CostRate) -> None: ...
    async def pending(self) -> BudgetUsageRows: ...
    async def usage(self) -> BudgetUsageRows: ...
    async def reconcile_pending(self, operation_id: str, actual_units: int, evidence: str) -> None: ...


class BillingCatalogPort(Protocol):
    async def contract(self, provider: str, model: str) -> ModelBillingContract: ...

    async def max_context(self, provider: str, model: str) -> int: ...
