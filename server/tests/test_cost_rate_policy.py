"""Input-only billing is explicit in Domain and mapped at external boundaries."""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, text

from domain.runtime.cost import BudgetError, CostRate
from infrastructure.budget.ledger import PostgresRedisBudget
from infrastructure.budget.runtime import BudgetRuntime
from presentation.api.routes.admin_config import CostRateRequest, add_cost_rate


@pytest.mark.parametrize("provider", ["deepinfra", "openrouter", "other"])
@pytest.mark.parametrize("input_only", [False, True])
def test_zero_output_requires_explicit_input_only_policy(provider: str, input_only: bool) -> None:
    rate = CostRate(provider, "m", 1, "tokens", Decimal(1), Decimal(0), 100, input_only)

    if input_only:
        assert rate.reserve(100, 0) == 100_000
    else:
        with pytest.raises(BudgetError):
            rate.reserve(100, 0)
        with pytest.raises(BudgetError, match="positive"):
            rate.reserve(100, 1)


@pytest.mark.parametrize("price", ["0", "-1", "NaN", "Infinity"])
def test_input_only_policy_still_rejects_invalid_input_tariffs(price: str) -> None:
    rate = CostRate("other", "m", 1, "tokens", Decimal(price), Decimal(0), 100, input_only=True)
    with pytest.raises(BudgetError):
        rate.reserve(100, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepinfra", "openrouter"])
async def test_api_maps_input_only_without_accepting_a_policy_override(provider: str) -> None:
    body = CostRateRequest.model_validate(
        {
            "provider": provider,
            "model": "m",
            "version": 1,
            "input_usd_per_million": "1",
            "output_usd_per_million": "0",
            "context_tokens": 100,
            "input_only": provider != "deepinfra",
        }
    )
    service = AsyncMock()
    await add_cost_rate(body, admin=Mock(), service=service)
    rate = service.add_rate.await_args.args[0]
    assert rate.input_only is (provider == "deepinfra")
    assert "input_only" not in CostRateRequest.model_json_schema()["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepinfra", "openrouter"])
async def test_stored_tariffs_derive_policy_without_a_database_schema_change(provider: str) -> None:
    engine = create_engine("sqlite://")
    try:
        with engine.connect() as connection:
            connection.execute(
                text("""
                CREATE TABLE cost_rates (
                    provider TEXT, model TEXT, version INTEGER, unit TEXT,
                    input_usd_per_million NUMERIC, output_usd_per_million NUMERIC,
                    context_tokens INTEGER
                )
            """)
            )
            connection.execute(
                text("""
                CREATE TABLE cost_usage (
                    provider TEXT, model TEXT, rate_version INTEGER,
                    actual_units INTEGER, reserved_units INTEGER
                )
            """)
            )
            connection.execute(
                text("INSERT INTO cost_rates VALUES (:provider, 'm', 1, 'tokens', 1, 0, 100)"),
                {"provider": provider},
            )
            session = AsyncMock()
            session.__aenter__.return_value = session
            session.execute.side_effect = connection.execute
            runtime = BudgetRuntime()  # type: ignore[no-untyped-call]
            runtime.session_factory = lambda: session
            runtime.catalog = AsyncMock()
            runtime.catalog.max_context.return_value = 100
            rate = await runtime.rate(provider, "m")  # type: ignore[no-untyped-call]
            assert bool(rate.input_only) is (provider == "deepinfra")
            rows = await PostgresRedisBudget(lambda: session, None).rates()  # type: ignore[no-untyped-call]
            assert bool(CostRate(**rows[0]).input_only) is (provider == "deepinfra")
    finally:
        engine.dispose()
