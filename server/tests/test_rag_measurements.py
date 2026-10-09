"""Benchmark timing and cost must describe the real calls, not just generation."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from domain.value_objects.chat_context import ChatContext
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.benchmark.answer_generators import BenchmarkAnswer, RagBenchmarkGenerator
from infrastructure.benchmark.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.token_usage import judge_usage, record_judge_usage
from infrastructure.ml.clients.managed_llm import ManagedLLM
from infrastructure.ml.usage_capture import active_llm_usage, record_model_response, summarize_usage


@pytest.mark.asyncio
async def test_all_managed_calls_keep_model_tokens_and_unknown_cost():
    async def invoke(*args, **kwargs):
        return AIMessage(
            content='answer',
            usage_metadata={'input_tokens': 11, 'output_tokens': 3, 'total_tokens': 14},
            response_metadata={'model_name': 'actual-model'},
        )

    records = []
    token = active_llm_usage.set(records)
    try:
        client = ManagedLLM(SimpleNamespace(ainvoke=invoke))
        await client.ainvoke('auxiliary')
        await client.ainvoke('generation')
    finally:
        active_llm_usage.reset(token)
    summary = summarize_usage(records)
    assert summary['calls'] == 2 and summary['input_tokens'] == 22
    assert records[0]['model'] == 'actual-model'
    assert summary['cost_usd'] is None and summary['cost_complete'] is False


def test_known_cost_and_missing_cost_are_not_summed_as_complete():
    records = [
        {'model': 'a', 'input_tokens': 10, 'output_tokens': 2, 'cost_usd': 0.01},
        {'model': 'b', 'input_tokens': 10, 'output_tokens': 2, 'cost_usd': None},
    ]
    assert summarize_usage(records)['cost_usd'] is None
    assert summarize_usage(records)['known_cost_usd'] == 0.01
    records[1]['cost_usd'] = 0.02
    assert summarize_usage(records)['cost_usd'] == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_rag_elapsed_includes_retrieval_and_usage_without_judge():
    async def invoke(**kwargs):
        await asyncio.sleep(0.02)  # Retrieval/selection time must belong to the RAG measurement.
        return SimpleNamespace(
            answer='answer',
            sources=[],
            input_tokens=None,
            output_tokens=None,
            ttft_sec=None,
            breadth=None,
            domain=None,
        )

    ctx = ChatContext(1, UserKind.INTERNAL, UserRole.ADMIN)
    result = await RagBenchmarkGenerator(SimpleNamespace(invoke=invoke), 8, 30).generate(
        {'question': 'q'}, ctx
    )
    assert result.rag_latency_sec >= 0.02
    assert result.llm_usage == []


@pytest.mark.asyncio
async def test_stream_usage_accumulates_chunks_and_captures_one_call():
    async def stream(*args, **kwargs):
        yield AIMessageChunk(content='answer', response_metadata={'model_name': 'stream-model'})
        yield AIMessageChunk(
            content='', usage_metadata={'input_tokens': 12, 'output_tokens': 4, 'total_tokens': 16}
        )

    records = []
    token = active_llm_usage.set(records)
    try:
        client = ManagedLLM(SimpleNamespace(astream=stream))
        chunks = [chunk async for chunk in client.astream('q')]
    finally:
        active_llm_usage.reset(token)
    assert len(chunks) == 2 and len(records) == 1
    assert records[0]['input_tokens'] == 12 and records[0]['model'] == 'stream-model'


@pytest.mark.asyncio
async def test_stream_preserves_repeated_model_and_generation_id():
    async def stream(*args, **kwargs):
        for content in ('first', 'last'):
            yield AIMessageChunk(
                content=content,
                id='gen-original',
                response_metadata={'model_name': 'stream-model', 'id': 'gen-original'},
            )

    records = []
    token = active_llm_usage.set(records)
    try:
        client = ManagedLLM(SimpleNamespace(astream=stream))
        async for _ in client.astream('q'):
            pass
    finally:
        active_llm_usage.reset(token)
    assert records[0]['model'] == 'stream-model'
    assert records[0]['response_id'] == 'gen-original'


@pytest.mark.asyncio
async def test_raw_stream_cost_survives_langchain_metadata_conversion():
    class RawStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def __aiter__(self):
            yield SimpleNamespace(
                model='actual-model',
                id='gen-cost',
                usage=SimpleNamespace(
                    model_dump=lambda: {
                        'prompt_tokens': 15,
                        'completion_tokens': 4,
                        'cost': 0.002,
                    }
                ),
            )

    async def create(**kwargs):
        return RawStream()

    client = SimpleNamespace(root_async_client=None, async_client=SimpleNamespace(create=create))

    async def stream(*args, **kwargs):
        raw = await client.async_client.create(stream=True)
        async with raw:
            async for _ in raw:
                # LangChain omits the provider's cost and generation ID here.
                yield AIMessageChunk(
                    content='answer',
                    usage_metadata={
                        'input_tokens': 15,
                        'output_tokens': 4,
                        'total_tokens': 19,
                    },
                )

    client.astream = stream
    records = []
    token = active_llm_usage.set(records)
    try:
        managed = ManagedLLM(client)
        async for _ in managed.astream('q'):
            pass
    finally:
        active_llm_usage.reset(token)
    assert len(records) == 1
    assert records[0]['cost_usd'] == 0.002
    assert records[0]['response_id'] == 'gen-cost'


@pytest.mark.asyncio
async def test_failed_rag_preserves_observed_usage_for_checkpoint():
    async def invoke(**kwargs):
        record_model_response(AIMessage(content='auxiliary', response_metadata={'model_name': 'model'}))
        raise RuntimeError('generation failed')

    records = []
    token = active_llm_usage.set(records)
    try:
        generator = RagBenchmarkGenerator(SimpleNamespace(invoke=invoke), 8, 30)
        with pytest.raises(RuntimeError, match='generation failed'):
            await generator.generate({'question': 'q'}, SimpleNamespace())
        assert len(records) == 1
    finally:
        active_llm_usage.reset(token)


def test_raw_usage_objects_keep_known_token_counts():
    records = []
    token = active_llm_usage.set(records)
    try:
        record_model_response(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=15, completion_tokens=4)))
    finally:
        active_llm_usage.reset(token)
    assert records[0]['input_tokens'] == 15 and records[0]['output_tokens'] == 4


@pytest.mark.asyncio
async def test_failed_judge_preserves_paid_responses_for_checkpoint(monkeypatch):
    async def generate(*args):
        return BenchmarkAnswer('answer', '', {}, None, None)

    async def judge(*args):
        record_judge_usage(
            SimpleNamespace(usage={'prompt_tokens': 15, 'completion_tokens': 4, 'cost': 0.002})
        )
        raise RuntimeError('judge validation failed')

    evaluator = BenchmarkCaseEvaluator(SimpleNamespace(generate=generate), 'model')
    monkeypatch.setattr(evaluator, 'judge_generated', judge)
    records = []
    token = judge_usage.set(records)
    try:
        with pytest.raises(RuntimeError, match='judge validation failed'):
            await evaluator.evaluate(0, {'question': 'q'}, 1, SimpleNamespace())
        assert len(records) == 1 and records[0]['cost_usd'] == 0.002
    finally:
        judge_usage.reset(token)
