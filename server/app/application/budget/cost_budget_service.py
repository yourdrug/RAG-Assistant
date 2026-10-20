"""Reserve before dispatch; ambiguous provider outcomes retain their reservation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Unpack

from domain.runtime.cost import BudgetError, usd_units


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from application.budget.dto import BudgetOperation
    from application.budget.ports import CostBudgetPort


class CostBudgetService:
    def __init__(self, ledger: CostBudgetPort):
        self.ledger = ledger

    @asynccontextmanager
    async def operation(self, **request: Unpack[BudgetOperation]) -> AsyncIterator[str]:
        policy = request["policy"]
        principal = request["principal"]
        operation_id = request["operation_id"]
        maximum_units = request["maximum_units"]

        if not policy.enabled:
            yield operation_id
            return

        if not principal or not (principal.startswith(("user:", "system:"))):
            msg = "Paid operation requires an explicit billing principal"
            raise BudgetError(msg)

        if maximum_units is None or maximum_units < 0:
            msg = "Unsupported or unknown billing; paid operation blocked"
            raise BudgetError(msg)

        hour, day = usd_units(policy.hourly_usd), usd_units(policy.daily_usd)

        if not hour or not day:
            msg = "Enabled cost limits must be positive"
            raise BudgetError(msg)

        admitted = await self.ledger.reserve(
            principal=principal,
            operation_id=operation_id,
            provider=request["provider"],
            model=request["model"],
            operation=request["operation"],
            maximum_units=maximum_units,
            rate_version=request["rate_version"],
            hourly_units=hour,
            daily_units=day,
            enforce_limits=policy.enabled,
        )

        if not admitted:
            msg = "Operation already reserved; reconcile before retrying"
            raise BudgetError(msg)

        # No automatic refund, including BaseException/cancellation.
        yield operation_id

    async def settle(self, principal: str, operation_id: str, actual_usd: str | None) -> None:
        if actual_usd is not None:
            await self.ledger.reconcile(principal, operation_id, usd_units(actual_usd))
