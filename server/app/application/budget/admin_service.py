"""Administration of immutable tariff versions and unresolved paid attempts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.value_objects.cost_budget import BudgetError, CostRate, usd_units


if TYPE_CHECKING:
    from application.budget.dto import BudgetUsageRows, CostRateRow, ModelBudgetStatus
    from application.budget.ports import BillingCatalogPort, BudgetAdminRepositoryPort


class BudgetAdminService:
    def __init__(self, repository: BudgetAdminRepositoryPort, catalog: BillingCatalogPort):
        self.repository = repository
        self.catalog = catalog

    async def rates(self) -> list[CostRateRow]:
        return await self.repository.rates()

    async def model_status(self, provider: str, model: str) -> ModelBudgetStatus:
        try:
            await self.check_model_tariff(provider, model)
        except BudgetError as exc:
            return {"provider": provider, "model": model, "ready": False, "reason": str(exc)}

        else:
            return {"provider": provider, "model": model, "ready": True, "reason": None}

    async def check_model_tariff(self, provider: str, model: str) -> None:
        context = await self.catalog.max_context(provider, model)
        rates = [
            CostRate(**row)
            for row in await self.repository.rates()
            if row["provider"] == provider and row["model"] == model
        ]

        if not rates:
            msg = "No published tariff"
            raise BudgetError(msg)

        rate = max(rates, key=lambda r: r.version)
        rate.reserve(context, 1)

        if await self.repository.rate_violated(provider, model, rate.version):
            msg = "This tariff exceeded its reservation bound; publish a verified new version"
            raise BudgetError(msg)

    async def add_rate(self, rate: CostRate) -> CostRate:
        rate.reserve(rate.context_tokens, 1)

        if rate.context_tokens < await self.catalog.max_context(rate.provider, rate.model):
            msg = "Tariff context is below the provider model bound"
            raise BudgetError(msg)

        await self.repository.add_rate(rate)
        return rate

    async def usage(self) -> BudgetUsageRows:
        return await self.repository.usage()

    async def pending(self) -> BudgetUsageRows:
        return await self.repository.pending()

    async def reconcile(self, operation_id: str, actual_usd: str, evidence: str) -> None:
        if not evidence.strip():
            msg = "Confirmed provider charge/evidence is required"
            raise BudgetError(msg)

        await self.repository.reconcile_pending(operation_id, usd_units(actual_usd), evidence)
