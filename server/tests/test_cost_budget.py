from decimal import Decimal
import pytest
from domain.value_objects.cost_budget import CostRate, BudgetError, usd_units
from application.budget import BudgetPolicy, CostBudgetService, billing_principal
from unittest.mock import AsyncMock
import asyncio
from types import SimpleNamespace
import httpx
from config import settings
from infrastructure.budget.runtime import BudgetRuntime, guard_paid_http_async
from infrastructure.ml.usage_capture import active_llm_usage


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
async def test_raw_attempt_is_reserved_once_and_usage_settled(monkeypatch):
    monkeypatch.setattr(settings, 'cost_rate_limit_enabled', True)
    runtime = BudgetRuntime()
    ledger = AsyncMock()
    ledger.reserve.return_value = True
    runtime.service = CostBudgetService(ledger)
    runtime.rate = AsyncMock(
        return_value=CostRate('openrouter', 'm', 1, 'tokens', Decimal('1'), Decimal('2'), 100)
    )
    identity = billing_principal.set('user:5')
    try:
        client = SimpleNamespace(
            openai_api_base='https://openrouter.ai/api/v1', model_name='m', max_tokens=10
        )
        async with runtime.llm_call(client, {}) as kwargs:
            ledger.reserve.assert_awaited_once()
            assert kwargs['extra_body']['provider']['max_price']['prompt'] == '1'
            request = httpx.Request(
                'POST',
                'https://openrouter.ai/api/v1/chat/completions',
                json={
                    **kwargs['extra_body'],
                    'model': 'm',
                    'max_tokens': 10,
                    'messages': [{'role': 'user', 'content': 'hello'}],
                },
            )
            await guard_paid_http_async(request)
            with pytest.raises(BudgetError):
                await guard_paid_http_async(request)
            active_llm_usage.get().append({'cost_usd': '0.0001'})
        ledger.reconcile.assert_awaited_once()
        assert ledger.reconcile.await_args.args[2] == 100_000
    finally:
        billing_principal.reset(identity)


@pytest.mark.asyncio
async def test_deepinfra_and_unreserved_raw_calls_fail_closed(monkeypatch):
    from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient

    monkeypatch.setattr(settings, 'cost_rate_limit_enabled', True)
    client = DeepInfraEmbeddingsClient('not-a-secret', 'https://api.deepinfra.com/v1/openai', 'm')
    try:
        with pytest.raises(BudgetError):
            await client.embed_query('hello')
        with pytest.raises(BudgetError):
            await guard_paid_http_async(
                httpx.Request('POST', 'https://openrouter.ai/api/v1/chat/completions')
            )
    finally:
        await client.close()


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
async def test_cancelled_sent_call_preserves_response_id_without_refund(monkeypatch):
    from infrastructure.budget.runtime import paid_attempt

    monkeypatch.setattr(settings, 'cost_rate_limit_enabled', True)
    runtime = BudgetRuntime()
    ledger = AsyncMock()
    ledger.reserve.return_value = True
    runtime.service = CostBudgetService(ledger)
    runtime.rate = AsyncMock(
        return_value=CostRate('openrouter', 'm', 1, 'tokens', Decimal('1'), Decimal('2'), 100)
    )
    principal_token = billing_principal.set('user:1')
    try:
        client = SimpleNamespace(
            openai_api_base='https://openrouter.ai/api/v1', model_name='m', max_tokens=10
        )
        with pytest.raises(asyncio.CancelledError):
            async with runtime.llm_call(client, {}) as kwargs:
                request = httpx.Request(
                    'POST',
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        **kwargs['extra_body'],
                        'model': 'm',
                        'max_tokens': 10,
                        'messages': [{'role': 'user', 'content': 'hello'}],
                    },
                )
                await guard_paid_http_async(request)
                active_llm_usage.get().append({'response_id': 'gen-example', 'cost_usd': None})
                raise asyncio.CancelledError
        ledger.reconcile.assert_not_called()
        ledger.remember_response.assert_awaited_once()
        assert ledger.remember_response.await_args.args[2] == 'gen-example'
        assert paid_attempt.get() is None
    finally:
        billing_principal.reset(principal_token)


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


@pytest.mark.asyncio
async def test_runtime_worker_retry_reuses_reservation_after_ambiguous_timeout(monkeypatch):
    from infrastructure.budget.job_context import bill_job_to_initiator
    from infrastructure.budget.runtime import paid_attempt

    monkeypatch.setattr(settings, 'cost_rate_limit_enabled', True)
    runtime = BudgetRuntime()
    operations = set()
    ids = []

    async def reserve(**kwargs):
        operation_id = kwargs['operation_id']
        ids.append(operation_id)
        if operation_id in operations:
            return False
        operations.add(operation_id)
        return True

    ledger = AsyncMock()
    ledger.reserve.side_effect = reserve
    runtime.service = CostBudgetService(ledger)
    runtime.rate = AsyncMock(
        return_value=CostRate('openrouter', 'm', 1, 'tokens', Decimal('1'), Decimal('2'), 100)
    )
    client = SimpleNamespace(openai_api_base='https://openrouter.ai/api/v1', model_name='m', max_tokens=10)
    dispatches = []

    @bill_job_to_initiator
    async def action(ctx):
        async with runtime.llm_call(client, {}):
            paid_attempt.get()['sent'] = True
            dispatches.append(True)
            raise TimeoutError('provider outcome unknown')

    with pytest.raises(TimeoutError):
        await action({'job_id': 'stable-job', 'job_try': 1}, cost_principal='user:1')
    with pytest.raises(BudgetError, match='already reserved'):
        await action({'job_id': 'stable-job', 'job_try': 2}, cost_principal='user:1')
    assert len(set(ids)) == 1
    assert len(dispatches) == 1
    ledger.reconcile.assert_not_called()
    with pytest.raises(TimeoutError):
        await action({'job_id': 'different-job'}, cost_principal='user:1')
    assert len(set(ids)) == 2


@pytest.mark.asyncio
async def test_worker_operation_ids_follow_requests_when_parallel_order_changes(monkeypatch):
    from infrastructure.budget.job_context import bill_job_to_initiator

    monkeypatch.setattr(settings, 'cost_rate_limit_enabled', True)
    runtime = BudgetRuntime()
    ledger = AsyncMock()
    runtime.service = CostBudgetService(ledger)
    runtime.rate = AsyncMock(
        return_value=CostRate('openrouter', 'm', 1, 'tokens', Decimal('1'), Decimal('2'), 100)
    )
    client = SimpleNamespace(openai_api_base='https://openrouter.ai/api/v1', model_name='m', max_tokens=10)
    client.ainvoke = AsyncMock(return_value=SimpleNamespace(content='answer'))
    observed = {}

    @bill_job_to_initiator
    async def action(ctx, requests):
        for request in requests:
            async with runtime.llm_call(client, {}, input=request):
                pass
            observed[(ctx['job_try'], request)] = ledger.reserve.await_args.kwargs['operation_id']

    await action({'job_id': 'job', 'job_try': 1}, ['a', 'b'], cost_principal='user:1')
    await action({'job_id': 'job', 'job_try': 2}, ['b', 'a'], cost_principal='user:1')
    assert observed[(1, 'a')] == observed[(2, 'a')]
    assert observed[(1, 'b')] == observed[(2, 'b')]
    assert observed[(1, 'a')] != observed[(1, 'b')]
