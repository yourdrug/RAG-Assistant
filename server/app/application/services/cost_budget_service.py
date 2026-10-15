"""Reserve before dispatch; ambiguous provider outcomes retain their reservation."""

from contextlib import asynccontextmanager
from dataclasses import dataclass

from application.ports.cost_budget import CostBudgetPort
from domain.value_objects.cost_budget import BudgetError, usd_units


@dataclass(frozen=True)
class BudgetPolicy:
    enabled: bool = False
    hourly_usd: str = '1.0'
    daily_usd: str = '5.0'


class CostBudgetService:
    def __init__(self, ledger: CostBudgetPort):
        self.ledger = ledger

    @asynccontextmanager
    async def operation(
        self,
        *,
        policy: BudgetPolicy,
        principal: str | None,
        operation_id: str,
        provider: str,
        model: str,
        operation: str,
        maximum_units: int | None,
        rate_version: int,
    ):
        if not policy.enabled:
            yield operation_id
            return
        if not principal or not (principal.startswith('user:') or principal.startswith('system:')):
            raise BudgetError('Paid operation requires an explicit billing principal')
        if maximum_units is None or maximum_units < 0:
            raise BudgetError('Unsupported or unknown billing; paid operation blocked')
        hour, day = usd_units(policy.hourly_usd), usd_units(policy.daily_usd)
        if policy.enabled and (not hour or not day):
            raise BudgetError('Enabled cost limits must be positive')
        admitted = await self.ledger.reserve(
            principal=principal,
            operation_id=operation_id,
            provider=provider,
            model=model,
            operation=operation,
            maximum_units=maximum_units,
            rate_version=rate_version,
            hourly_units=hour,
            daily_units=day,
            enforce_limits=policy.enabled,
        )
        if not admitted:
            raise BudgetError('Operation already reserved; reconcile before retrying')
        # No automatic refund, including BaseException/cancellation.
        yield operation_id

    async def settle(self, principal: str, operation_id: str, actual_usd: str | None):
        if actual_usd is not None:
            await self.ledger.reconcile(principal, operation_id, usd_units(actual_usd))
