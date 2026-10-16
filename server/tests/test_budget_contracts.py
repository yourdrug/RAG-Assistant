"""Compatibility and admission boundaries after the budget move."""

import importlib
from unittest.mock import AsyncMock

import pytest

from application.budget import BudgetAdminService, BudgetPolicy, CostBudgetService
from domain.value_objects.cost_budget import BudgetError


@pytest.mark.parametrize(
    "legacy,names",
    [
        ("application.ports.billing_context", ("billing_principal",)),
        (
            "application.ports.cost_budget",
            ("BillingCatalogPort", "BudgetAdminRepositoryPort", "CostBudgetPort"),
        ),
        ("application.services.cost_budget_service", ("BudgetPolicy", "CostBudgetService")),
        ("application.services.budget_admin_service", ("BudgetAdminService",)),
    ],
)
def test_legacy_exports_share_canonical_objects(legacy: str, names: tuple[str, ...]) -> None:
    old = importlib.import_module(legacy)
    canonical = importlib.import_module("application.budget")

    for name in names:
        assert getattr(old, name) is getattr(canonical, name)


@pytest.mark.asyncio
async def test_disabled_policy_bypasses_billing_validation() -> None:
    ledger = AsyncMock()
    service = CostBudgetService(ledger)
    async with service.operation(
        policy=BudgetPolicy(),
        principal=None,
        operation_id="disabled",
        provider="unknown",
        model="unknown",
        operation="llm",
        maximum_units=None,
        rate_version=0,
    ) as operation_id:
        assert operation_id == "disabled"
    ledger.reserve.assert_not_awaited()
    ledger.reconcile.assert_not_awaited()


@pytest.mark.parametrize("policy", [BudgetPolicy(True, "0", "5"), BudgetPolicy(True, "1", "0")])
@pytest.mark.asyncio
async def test_zero_enabled_limit_blocks_before_reservation(policy: BudgetPolicy) -> None:
    ledger = AsyncMock()
    with pytest.raises(BudgetError, match="must be positive"):
        async with CostBudgetService(ledger).operation(
            policy=policy,
            principal="user:1",
            operation_id="op",
            provider="openrouter",
            model="m",
            operation="llm",
            maximum_units=1,
            rate_version=1,
        ):
            pytest.fail("zero quota must block dispatch")
    ledger.reserve.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_without_tariff_is_unready() -> None:
    repository, catalog = AsyncMock(), AsyncMock()
    repository.rates.return_value = []
    catalog.max_context.return_value = 100
    status = await BudgetAdminService(repository, catalog).model_status("openrouter", "m")
    assert status == {
        "provider": "openrouter",
        "model": "m",
        "ready": False,
        "reason": "No published tariff",
    }
    repository.rate_violated.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconciliation_requires_evidence_and_converts_usd() -> None:
    repository = AsyncMock()
    service = BudgetAdminService(repository, AsyncMock())
    with pytest.raises(BudgetError, match="evidence"):
        await service.reconcile("op", "0.1", "  ")
    repository.reconcile_pending.assert_not_awaited()
    await service.reconcile("op", "0.1", "provider receipt")
    repository.reconcile_pending.assert_awaited_once_with("op", 100_000_000, "provider receipt")
