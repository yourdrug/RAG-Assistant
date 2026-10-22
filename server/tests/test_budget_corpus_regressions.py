"""Regressions for zero tariffs and corpus switches during evaluations."""

from infrastructure.benchmark.sweep import candidates as sweep_candidates


from infrastructure.worker import runtime

import asyncio
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
from qdrant_client.models import Filter

import pytest

from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from fakes import FakeUnitOfWorkFactory
from pydantic import ValidationError

from application.budget import BudgetAdminService, BudgetPolicy, CostBudgetService
from domain.benchmark.sweep import BenchmarkSweep
from domain.runtime.cost import BudgetError, CostRate
from infrastructure.benchmark.evaluation import runner
from infrastructure.benchmark.corpus import corpus_metadata
from composition.service_providers import build_sweep_engine
from infrastructure.budget.provider_catalog import OpenRouterBillingCatalog
from infrastructure.budget.runtime import BudgetRuntime
from infrastructure.budget.transport import PaidAsyncTransport, prepare_inference_body
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.worker import tasks
from presentation.api.routes.admin_config import CostRateRequest, CostRateResponse


@pytest.mark.parametrize(
    'provider,inputs,outputs',
    [
        ('deepinfra', '0', '0'),
        ('deepinfra', '0', '1'),
        ('openrouter', '0', '1'),
        ('openrouter', '1', '0'),
    ],
)
@pytest.mark.asyncio
async def test_zero_paid_tariff_cannot_be_published(provider, inputs, outputs):
    rate = CostRate(provider, 'm', 1, 'tokens', Decimal(inputs), Decimal(outputs), 32768)
    repository = AsyncMock()
    service = BudgetAdminService(repository, AsyncMock())
    with pytest.raises(BudgetError, match='positive'):
        await service.add_rate(rate)
    repository.add_rate.assert_not_awaited()


def test_api_rejects_zero_input_but_lists_legacy_tariff():
    data = {
        'provider': 'deepinfra',
        'model': 'm',
        'version': 1,
        'unit': 'tokens',
        'input_usd_per_million': '0',
        'output_usd_per_million': '0',
        'context_tokens': 32768,
    }
    with pytest.raises(ValidationError):
        CostRateRequest(**data)
    assert CostRateResponse(**data).input_usd_per_million == 0


@pytest.mark.asyncio
async def test_stored_zero_tariff_never_dispatches():
    runtime = BudgetRuntime()
    runtime.catalog = OpenRouterBillingCatalog()
    runtime.service = CostBudgetService(AsyncMock())
    runtime.policy_snapshot = AsyncMock(return_value=BudgetPolicy(True))
    runtime.rate = AsyncMock(
        return_value=CostRate(
            'deepinfra',
            'Qwen/Qwen3-Reranker-0.6B',
            1,
            'tokens',
            Decimal('0'),
            Decimal('0'),
            32768,
            input_only=True,
        )
    )
    send = Mock(return_value=httpx.Response(200))
    async with httpx.AsyncClient(
        transport=PaidAsyncTransport(
            runtime=runtime,
            transport=httpx.MockTransport(send),
        )
    ) as client:
        with pytest.raises(BudgetError, match='positive'):
            await client.post(
                'https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-0.6B',
                json={'queries': ['q'], 'documents': ['d']},
            )
    send.assert_not_called()
    runtime.service.ledger.reserve.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('model', ['Qwen/Qwen3-Reranker-0.6B', 'Qwen/Qwen3-Reranker-4B'])
async def test_reranker_contract_reserves_each_pair(model):
    contract = await OpenRouterBillingCatalog().contract('deepinfra', model)
    rate = CostRate('deepinfra', model, 1, 'tokens', Decimal('0.01'), Decimal('0'), 32768, input_only=True)
    assert prepare_inference_body({'queries': ['q'], 'documents': ['d1', 'd2']}, contract, rate) == 655360


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'model,price,ready',
    [
        ('Qwen/Qwen3-Reranker-0.6B', '0.01', True),
        ('Qwen/Qwen3-Reranker-0.6B', '0', False),
        ('BAAI/bge-reranker-v2-m3', '0.01', False),
    ],
)
async def test_active_model_status_reports_unsupported_or_invalid_tariffs(model, price, ready):
    repository = AsyncMock()
    repository.rates.return_value = [
        vars(
            CostRate(
                'deepinfra',
                model,
                1,
                'tokens',
                Decimal(price),
                Decimal('0'),
                32768,
                input_only=True,
            )
        )
    ]
    repository.rate_violated.return_value = False
    status = await BudgetAdminService(repository, OpenRouterBillingCatalog()).model_status('deepinfra', model)
    assert status['ready'] is ready
    if not ready:
        assert status['reason']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'versions,violated,ready', [([1], False, True), ([1], True, False), ([1, 2], True, True)]
)
async def test_active_model_status_checks_violation_of_latest_tariff(versions, violated, ready):
    model = 'Qwen/Qwen3-Reranker-0.6B'
    repository = AsyncMock()
    repository.rates.return_value = [
        vars(
            CostRate(
                'deepinfra', model, version, 'tokens', Decimal('0.01'), Decimal('0'), 32768, input_only=True
            )
        )
        for version in versions
    ]
    repository.rate_violated.side_effect = lambda provider, name, version: violated and version == 1
    status = await BudgetAdminService(repository, OpenRouterBillingCatalog()).model_status('deepinfra', model)
    assert status['ready'] is ready
    repository.rate_violated.assert_awaited_once_with('deepinfra', model, max(versions))
    if not ready:
        assert 'exceeded its reservation bound' in status['reason']


def corpus_pair():
    registry = MLClientRegistry()
    snapshot = CorpusSnapshot('http://qdrant:6333', 'docs', 'v1', 'identity', 'generation1')
    old = BoundQdrantClient(Mock(), snapshot)
    new = BoundQdrantClient(Mock(), replace(snapshot, physical_collection='v2', generation='generation2'))
    registry.install_corpus(old, None)
    return registry, old, new


@pytest.mark.asyncio
async def test_benchmark_pins_all_parallel_and_queued_questions(monkeypatch, tmp_path):
    registry, old, new = corpus_pair()
    observed = []
    switched = asyncio.Event()

    async def evaluate(self, idx, question, run_idx, ctx, **kwargs):
        if idx == 1:
            registry.install_corpus(new, None)
            switched.set()
        await switched.wait()
        await asyncio.sleep(0)
        observed.append(registry.qdrant_client())
        assert corpus_metadata(registry.corpus_snapshot)['corpus_generation'] == 'generation1'
        old.client.close.assert_not_called()
        return {
            'id': idx,
            'question': question['question'],
            'generator_metrics': {},
            'retriever_metrics': {},
            'latency_sec': 1,
        }

    monkeypatch.setattr(runner.BenchmarkCaseEvaluator, 'evaluate', evaluate)
    monkeypatch.setattr(runner.progress, 'log_question_result', lambda *args: None)
    monkeypatch.setattr(runner.progress, 'log_summary', lambda *args: None)
    try:
        await runner.run_benchmark_async(
            [{'question': 'q', 'annotations': {'expected_refusal': True}}] * 3,
            1,
            'judge',
            max_concurrent=2,
            rag_service=SimpleNamespace(),
            corpus_resource=registry,
        )
        assert observed == [old] * 3
        assert list(tmp_path.iterdir()) == []
        await old.aclose()
        old.client.close.assert_called_once()
        assert registry.qdrant_client() is new
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_candidate_cache_keeps_client_across_switch(monkeypatch):
    registry, old, new = corpus_pair()
    old.client.search.return_value = []
    registry._embeddings = Mock()
    registry._embeddings.embed_query_sync.return_value = [1.0]

    def search(*args, **kwargs):
        registry.install_corpus(new, None)
        return []
    old.client.search.side_effect = search
    try:
        await sweep_candidates.cache_candidates(
            [{'question': 'alpha'}, {'question': 'beta'}], 1, ml_clients=registry,
            access_filter=Filter(), visibility_conditions=[object()], user_id=0, user_group_ids=[],
        )
        assert old.client.search.call_count == 4
        new.client.search.assert_not_called()
        assert all(call.kwargs['collection_name'] == 'v1' for call in old.client.search.call_args_list)
    finally:
        await old.aclose()
        await registry.close()


@pytest.mark.asyncio
async def test_sweep_pins_preparation_and_rejects_resume_on_changed_corpus(tmp_path, monkeypatch):
    registry, old, new = corpus_pair()
    factory = FakeUnitOfWorkFactory()
    def checkpoints(sweep_id):
        return DatabaseBenchmarkCheckpoints(factory, sweep_id)
    engine = build_sweep_engine(None, ml_clients=registry, checkpoint_factory=checkpoints)
    engine.load_questions = AsyncMock(return_value=[{'question': 'q', 'source_hint': 'law.pdf'}])

    async def phase_a(*args):
        registry.install_corpus(new, None)
        return [{'config': {'top_k': 1}, 'composite_score': 1}]

    async def phase_b(*args, **kwargs):
        assert registry.qdrant_client() is old
        assert kwargs['run'].corpus == corpus_metadata(old.snapshot)
        old.client.close.assert_not_called()
        return []

    engine.evaluate_phase_a = phase_a
    engine.run_phase_b = phase_b
    sweep = BenchmarkSweep(id=123, search_space={'top_k': {'values': [1]}}, objective_weights={'hit_rate': 1})
    try:
        await engine.run_sweep(sweep, 'judge')
        with pytest.raises(ValueError, match='corpus changed'):
            await engine.run_sweep(sweep, 'judge')
    finally:
        await old.aclose()
        await registry.close()


@pytest.mark.asyncio
async def test_worker_report_uses_pinned_corpus_after_switch(monkeypatch):
    registry, old, new = corpus_pair()
    service = SimpleNamespace(get_job_result=AsyncMock(return_value=None), record_result=AsyncMock())
    service.record_result.return_value = SimpleNamespace(id=None)
    container = SimpleNamespace(
        infrastructure=SimpleNamespace(
            db=SimpleNamespace(uow_factory=object()), ml=SimpleNamespace(ml_clients=registry)
        ),
        application=SimpleNamespace(rag_service=SimpleNamespace()),
    )
    monkeypatch.setattr(tasks, 'load_benchmark_questions', AsyncMock(return_value=[{'question': 'q'}]))

    async def evaluate(**kwargs):
        registry.install_corpus(new, None)
        return {"results": [{}], "total_time_sec": 1}

    async def tracked(factory, job_id, action, **kwargs):
        await action()

    container.infrastructure.ml = SimpleNamespace(
        ml_clients=registry, benchmark_service=SimpleNamespace(run=evaluate)
    )
    monkeypatch.setattr(runtime, 'run_tracked_job', tracked)
    try:
        await tasks.run_benchmark(
            {'container': container, 'benchmark_results': service},
            dataset='main', top_k=1, judge_model='judge', job_id=1
        )
        report = service.record_result.call_args.args[0]
        assert report.config_json['collection_name'] == 'v1'
        assert report.config_json['corpus_generation'] == 'generation1'
        assert report.config_json['embedding_identity'] == 'identity'
    finally:
        await old.aclose()
        await registry.close()
