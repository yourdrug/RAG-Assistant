"""Regression coverage for source labels across all benchmark paths."""

from unittest.mock import AsyncMock

import pytest
from langchain.schema import Document

from domain.entities.benchmark_sweep import BenchmarkSweep
from infrastructure.benchmark.case_metrics import compute_retriever_metrics_from_sources
from infrastructure.benchmark.metrics import compute_retriever_metrics, compute_summary_metrics
from infrastructure.benchmark.sweep_engine import SweepEngine
from infrastructure.benchmark.sweep_scoring import score_config_cheap


def sweep_score(questions, filenames):
    docs = [Document(page_content=name, metadata={'filename': name}) for name in filenames]
    return score_config_cheap(
        {'top_k': 10, 'fetch_k': 10, 'dense_weight': 1, 'sparse_weight': 0},
        questions,
        {q['question']: [(str(i), 0.8, doc) for i, doc in enumerate(docs)] for q in questions},
        {},
        {str(i): doc for i, doc in enumerate(docs)},
        {"hit_rate": 1},
    )


@pytest.fixture(params=['documents', 'rag_sources', 'sweep'])
def evaluate(request):
    def score(hint, filenames):
        if request.param == 'documents':
            docs = [Document(page_content=name, metadata={'source': name}) for name in filenames]
            return compute_retriever_metrics([(doc, 0.8) for doc in docs], hint)
        if request.param == 'rag_sources':
            return compute_retriever_metrics_from_sources(
                [{'source': name, 'max_score': 0.8} for name in filenames], hint
            )
        result = sweep_score([{'question': 'q', 'source_hint': hint}], filenames)
        return {'hit_rate': result['avg_hit_rate'], 'mrr': result['avg_mrr']}

    return request.param, score


@pytest.mark.parametrize(
    ('hint', 'filenames', 'hit', 'mrr'),
    [
        ('FIRST.pdf; SECOND.pdf', ['other.pdf', 'second.pdf'], 1, 0.5),
        ('first.pdf; second.pdf', ['first.pdf', 'second.pdf'], 1, 1.0),
        ('first.pdf; second.pdf', ['second.pdf', 'first.pdf'], 1, 1.0),
        ('first.pdf; second.pdf', ['other.pdf'], 0, 0.0),
        ('first.pdf; second.pdf', [], 0, 0.0),
        (' ; first.pdf; ; second.pdf ; ', ['other.pdf', 'second.pdf'], 1, 0.5),
        ('missing.pdf;', ['other.pdf'], 0, 0.0),
    ],
)
def test_multiple_sources_match_any_hint_at_first_relevant_rank(evaluate, hint, filenames, hit, mrr):
    _, score = evaluate
    result = score(hint, filenames)
    assert result['hit_rate'] == hit
    assert result['mrr'] == mrr


@pytest.mark.parametrize('hint', [None, '', '  \t\n', ';', ' ; ; '])
@pytest.mark.parametrize('filenames', [[], ['other.pdf']])
def test_unlabelled_questions_have_no_retrieval_metrics(evaluate, hint, filenames):
    kind, score = evaluate
    result = score(hint, filenames)
    expected = None
    assert result['hit_rate'] == expected
    assert result['mrr'] == expected
    if kind != 'sweep':
        assert result['retrieved_sources'] == filenames
        assert result['avg_similarity'] == (0.8 if filenames else 0.0)


def test_sweep_excludes_blank_hints_from_average():
    result = sweep_score(
        [
            {'question': 'blank', 'source_hint': ''},
            {'question': 'miss', 'source_hint': 'missing.pdf'},
        ],
        ['other.pdf'],
    )
    assert result['avg_hit_rate'] == 0
    assert result['avg_mrr'] == 0


def test_summary_excludes_blank_hints_from_average():
    results = [
        {
            'retriever_metrics': compute_retriever_metrics_from_sources(
                [{'source': 'other.pdf', 'max_score': 0.8}], hint
            ),
            'generator_metrics': {'faithfulness': 8, 'relevancy': 8, 'correctness': None},
            'latency_sec': 1,
        }
        for hint in ['', 'missing.pdf']
    ]
    summary = compute_summary_metrics(results)
    assert summary['hit_rate'] == 0
    assert summary['avg_mrr'] == 0
    assert summary['total_questions'] == 2


@pytest.mark.asyncio
async def test_sweep_does_not_fetch_candidates_for_blank_hints(monkeypatch):
    engine = SweepEngine(uow_factory=None)
    valid = {'question': 'valid', 'source_hint': 'first.pdf; second.pdf'}
    monkeypatch.setattr(
        engine,
        'load_questions',
        AsyncMock(
            return_value=[
                {'question': 'empty', 'source_hint': ''},
                {'question': 'spaces', 'source_hint': '  '},
                {'question': 'separators', 'source_hint': '; ;'},
                {'question': 'none', 'source_hint': None},
                valid,
            ]
        ),
    )
    cache = AsyncMock(return_value=({}, {}, {}))
    monkeypatch.setattr(engine, 'cache_candidates', cache)
    await engine.run_sweep(BenchmarkSweep(search_space={'top_k': {'values': [2]}}, top_n_llm=0))
    assert cache.call_args.args[0] == [valid]
