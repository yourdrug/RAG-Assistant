"""Durable admission contract for each physical paid provider attempt."""

from typing import Protocol

from domain.value_objects.cost_budget import ModelBillingContract


class CostBudgetPort(Protocol):
    async def reserve(
        self,
        *,
        principal: str,
        operation_id: str,
        provider: str,
        model: str,
        operation: str,
        maximum_units: int,
        rate_version: int,
        hourly_units: int,
        daily_units: int,
        enforce_limits: bool = True,
        request_fingerprint: str | None = None,
    ) -> bool:
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
        billable_units: dict | None = None,
        cost_source: str | None = None,
    ) -> None:
        """Idempotently settle usage; unknown outcomes remain reserved."""
        ...


class BudgetAdminRepositoryPort(Protocol):
    async def rates(self) -> list: ...
    async def rate_violated(self, provider: str, model: str, version: int) -> bool: ...
    async def add_rate(self, rate) -> None: ...
    async def pending(self) -> list: ...
    async def usage(self) -> list: ...
    async def reconcile_pending(self, operation_id: str, actual_units: int, evidence: str) -> None: ...


class BillingCatalogPort(Protocol):
    async def contract(self, provider: str, model: str) -> ModelBillingContract: ...

    async def max_context(self, provider: str, model: str) -> int: ...
