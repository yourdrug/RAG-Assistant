from decimal import Decimal
import pytest
from domain.runtime.cost import CostRate, BudgetError, usd_units
from application.budget import BudgetPolicy, CostBudgetService, billing_principal
from unittest.mock import AsyncMock
import asyncio
from config import settings


def test_money_rounds_up_without_float_arithmetic():
    assert usd_units('0.000000001') == 1
    assert usd_units('1.000000001') == 1_000_000_001
    for value in ('NaN', 'Infinity', '-1'):
        with pytest.raises(BudgetError):
            usd_units(value)


def test_versioned_rate_reserves_bounded_usage():
    rate = CostRate('openrouter', 'model', 3, 'tokens', Decimal('0.1'), Decimal('0.2'), 1000)
    assert rate.reserve(100, 10) == 12_000
    with pytest.raises(BudgetError):
        rate.reserve(1001, 10)
    with pytest.raises(BudgetError):
        rate.reserve(100, 0)


def test_unknown_or_non_token_billing_is_not_free():
    with pytest.raises(BudgetError):
        CostRate('deepinfra', 'model', 1, 'unknown', Decimal('0'), Decimal('0'), 100).reserve(1, 1)


@pytest.mark.asyncio
async def test_cancelled_call_retains_durable_reservation_and_retry_is_blocked():
    ledger = AsyncMock()
    ledger.reserve.side_effect = [True, False]
    service = CostBudgetService(ledger)
    args = {
        'policy': BudgetPolicy(True),
        'principal': 'user:1',
        'operation_id': 'op',
        'provider': 'openrouter',
        'model': 'm',
        'operation': 'chat',
        'maximum_units': 100,
        'rate_version': 1,
    }
    with pytest.raises(asyncio.CancelledError):
        async with service.operation(**args):
            raise asyncio.CancelledError
    ledger.reconcile.assert_not_called()
    with pytest.raises(BudgetError, match='already reserved'):
        async with service.operation(**args):
            pytest.fail('must not dispatch twice')


@pytest.mark.asyncio
async def test_unknown_price_and_missing_principal_block_before_dispatch():
    ledger = AsyncMock()
    service = CostBudgetService(ledger)
    for principal, maximum in [(None, 1), ('user:1', None)]:
        with pytest.raises(BudgetError):
            async with service.operation(
                policy=BudgetPolicy(True),
                principal=principal,
                operation_id='op',
                provider='x',
                model='m',
                operation='rerank',
                maximum_units=maximum,
                rate_version=1,
            ):
                pytest.fail('must not dispatch')
    ledger.reserve.assert_not_called()


@pytest.mark.asyncio
async def test_backend_error_is_fail_closed():
    ledger = AsyncMock()
    ledger.reserve.side_effect = ConnectionError('offline')
    with pytest.raises(ConnectionError):
        async with CostBudgetService(ledger).operation(
            policy=BudgetPolicy(True),
            principal='user:1',
            operation_id='op',
            provider='x',
            model='m',
            operation='embedding',
            maximum_units=1,
            rate_version=1,
        ):
            pytest.fail('must not dispatch')


@pytest.mark.asyncio
async def test_deepinfra_without_backend_fails_closed(monkeypatch):
    from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient

    monkeypatch.setattr(settings, 'cost_rate_limit_enabled', True)
    with pytest.raises(BudgetError, match='Budget runtime must be supplied for paid clients'):
        DeepInfraEmbeddingsClient('not-a-secret', 'https://api.deepinfra.com/v1/openai', 'm')


@pytest.mark.asyncio
async def test_redis_budget_gate_rejects_lowered_quota_and_recovers_after_restart():
    import fakeredis.aioredis
    from infrastructure.budget.ledger import ADMIT

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        assert await redis.eval(ADMIT, 1, 'budget', 90, 90, 10, 100, 500, 'op1') == 1
        assert await redis.eval(ADMIT, 1, 'budget', 100, 100, 1, 99, 500, 'op2') == 0
        await redis.flushall()
        # Committed PostgreSQL totals are supplied on every admission, including after restart.
        assert await redis.eval(ADMIT, 1, 'budget', 100, 100, 1, 100, 500, 'op2') == 0
        assert await redis.eval(ADMIT, 1, 'budget', 100, 100, 1, 101, 500, 'op2') == 1
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_publishing_tariff_rejects_understated_provider_context():
    from application.budget import BudgetAdminService

    repository, catalog = AsyncMock(), AsyncMock()
    catalog.max_context.return_value = 100
    service = BudgetAdminService(repository, catalog)
    rate = CostRate('openrouter', 'm', 1, 'tokens', Decimal('1'), Decimal('2'), 99)
    with pytest.raises(BudgetError, match='below'):
        await service.add_rate(rate)
    repository.add_rate.assert_not_called()
    valid = CostRate('openrouter', 'm', 2, 'tokens', Decimal('1'), Decimal('2'), 100)
    await service.add_rate(valid)
    repository.add_rate.assert_awaited_once_with(valid)


@pytest.mark.asyncio
async def test_worker_billing_identity_is_inherited_and_restored():
    from infrastructure.budget.job_context import bill_job_to_initiator

    observed = []

    @bill_job_to_initiator
    async def action(ctx):
        observed.append(billing_principal.get())

    token = billing_principal.set('system:worker')
    try:
        await action({}, cost_principal='user:123')
        assert observed == ['user:123']
        assert billing_principal.get() == 'system:worker'
    finally:
        billing_principal.reset(token)
