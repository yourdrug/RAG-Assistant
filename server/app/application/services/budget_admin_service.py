"""Administration of immutable tariff versions and unresolved paid attempts."""

from domain.value_objects.cost_budget import BudgetError, CostRate, usd_units
from application.ports.cost_budget import BudgetAdminRepositoryPort, BillingCatalogPort


class BudgetAdminService:
    def __init__(self, repository: BudgetAdminRepositoryPort, catalog: BillingCatalogPort):
        self.repository = repository
        self.catalog = catalog

    async def rates(self):
        return await self.repository.rates()

    async def add_rate(self, rate: CostRate):
        rate.reserve(rate.context_tokens, 1)
        if rate.context_tokens < await self.catalog.max_context(rate.provider, rate.model):
            raise BudgetError("Tariff context is below the provider model bound")
        await self.repository.add_rate(rate)
        return rate

    async def usage(self):
        return await self.repository.usage()

    async def pending(self):
        return await self.repository.pending()

    async def reconcile(self, operation_id: str, actual_usd: str, evidence: str):
        if not evidence.strip():
            raise BudgetError('Confirmed provider charge/evidence is required')
        await self.repository.reconcile_pending(operation_id, usd_units(actual_usd), evidence)
