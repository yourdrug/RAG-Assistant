"""Public workflow contracts captured before splitting the implementation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import _settings_overrides, get_setting
from domain.entities.benchmark_sweep import BenchmarkSweep
from infrastructure.benchmark import judge, metrics, persistence
from infrastructure.benchmark.runner import run_benchmark_async
from infrastructure.benchmark.sweep_engine import SweepEngine, SweepCancelled
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository


@pytest.fixture
def benchmark_io(monkeypatch):
    questions = [{"id": "first", "question": "question", "source_hint": "report"}]
    score = AsyncMock(return_value={"faithfulness": 8, "relevancy": 7, "correctness": None})
    monkeypatch.setattr(judge, "judge_answer_async", score)
    monkeypatch.setattr(
        metrics, "compute_context_precision_recall", lambda *a, **kw: {"context_precision": 6}
    )
    monkeypatch.setattr(metrics, "_estimate_cost_usd", lambda *a: 0.25)
    saved = MagicMock()
    monkeypatch.setattr(persistence, "save_results", saved)
    monkeypatch.setattr(persistence, "log_question_result", lambda *a: None)
    monkeypatch.setattr(persistence, "log_summary", lambda *a: None)
    return questions, saved, score


@pytest.mark.asyncio
async def test_full_benchmark_preserves_results_repeated_runs_and_cache_scope(benchmark_io):
    questions, saved, score = benchmark_io
    contexts = []

    async def invoke(question, history, ctx):
        from langchain.schema import Document
        from infrastructure.ml.rag.benchmark_evidence import capture_prompt

        assert get_setting("cache_enabled") is False
        contexts.append(ctx)
        docs = [(Document(page_content="context", metadata={"source": "report.pdf"}), 0.9)]
        capture_prompt(docs, docs, "context")
        return SimpleNamespace(
            answer="answer",
            input_tokens=20,
            output_tokens=10,
            ttft_sec=0.5,
            breadth="narrow",
            domain="general",
            sources=[{"source": "report.pdf", "max_score": 0.9, "content": "context"}],
        )

    previous = _settings_overrides.get()
    await run_benchmark_async(
        questions, "unused", 4, "judge", n_runs=2, rag_service=SimpleNamespace(invoke=invoke)
    )
    assert _settings_overrides.get() == previous
    results = saved.call_args.args[0]
    assert [r["run"] for r in results] == [1, 2]
    assert [r["id"] for r in results] == ["first", "first"]
    assert results[0]["retriever_metrics"]["hit_rate"] == 1
    assert results[0]["cost_usd"] == 0.25
    assert results[0]["input_tokens"] == 20
    assert all(c.user_role == "admin" and c.user_kind == "internal" for c in contexts)
    assert score.call_args.kwargs["context"] == "context"
    assert saved.call_args.kwargs["run_id"] == "all"


@pytest.mark.asyncio
async def test_benchmark_restores_cache_scope_and_propagates_pipeline_error(benchmark_io):
    questions, saved, _ = benchmark_io
    previous = _settings_overrides.get()
    rag = SimpleNamespace(invoke=AsyncMock(side_effect=RuntimeError("pipeline failed")))
    with pytest.raises(RuntimeError, match="pipeline failed"):
        await run_benchmark_async(questions, "unused", 4, "judge", rag_service=rag)
    assert _settings_overrides.get() == previous
    saved.assert_not_called()


@pytest.mark.asyncio
async def test_standalone_benchmark_keeps_acl_context_and_usage(benchmark_io, monkeypatch):
    from infrastructure.benchmark import retrieval

    questions, saved, score = benchmark_io
    docs = [(SimpleNamespace(page_content="context", metadata={"source": "report.pdf"}), 0.9)]
    retrieve = MagicMock(return_value=docs)
    monkeypatch.setattr(retrieval, "build_llm", lambda *a, **kw: object())
    monkeypatch.setattr(retrieval, "retrieve_with_scores_hybrid", retrieve)
    response = SimpleNamespace(
        response_metadata={"token_usage": {"prompt_tokens": 30, "completion_tokens": 5}}
    )
    monkeypatch.setattr(judge, "get_rag_answer_with_usage", lambda *a: ("answer", response))
    await run_benchmark_async(questions, "unused", 4, "judge", fetch_k=12)
    assert retrieve.call_args.args[1:3] == (4, 12)
    assert retrieve.call_args.kwargs["access_filter"] is not None
    assert retrieve.call_args.kwargs["visibility_conditions"]
    result = saved.call_args.args[0][0]
    assert result["input_tokens"] == 30 and result["output_tokens"] == 5
    assert result["breadth"] is None
    assert score.call_args.kwargs["context"] == "[1] report.pdf\ncontext"
    assert result["evidence"]["context"] == score.call_args.kwargs["context"]


@pytest.fixture
def grid_engine(monkeypatch):
    engine = SweepEngine(uow_factory=None)
    monkeypatch.setattr(
        engine, "load_questions", AsyncMock(return_value=[{"question": "q", "source_hint": "doc"}])
    )
    monkeypatch.setattr(engine, "cache_candidates", AsyncMock(return_value=({}, {}, {})))
    monkeypatch.setattr(engine, "score_config_cheap", lambda cfg, *a: {"composite_score": cfg["top_k"]})
    return engine


@pytest.mark.asyncio
async def test_grid_sweep_scores_sorts_and_reports_progress(grid_engine):
    progress = []
    sweep = BenchmarkSweep(search_space={"top_k": {"values": [2, 5]}}, top_n_llm=0)
    results = await grid_engine.run_sweep(sweep, progress_callback=lambda *args: progress.append(args))
    assert [r["config"]["top_k"] for r in results] == [5, 2]
    assert [(p[0], p[1]) for p in progress] == [(1, 2), (2, 2)]


@pytest.mark.asyncio
async def test_sweep_cancels_before_scoring(grid_engine, monkeypatch):
    score = MagicMock()
    monkeypatch.setattr(grid_engine, "score_config_cheap", score)
    sweep = BenchmarkSweep(search_space={"top_k": {"values": [2]}}, top_n_llm=0)
    with pytest.raises(SweepCancelled):
        await grid_engine.run_sweep(sweep, should_cancel=AsyncMock(return_value=True))
    score.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_restores_existing_overrides_when_full_benchmark_fails(grid_engine, monkeypatch):
    token = _settings_overrides.set({"cache_enabled": True, "retriever_top_k": 99})
    monkeypatch.setattr(
        grid_engine, "run_full_benchmark", AsyncMock(side_effect=RuntimeError("judge failed"))
    )
    try:
        sweep = BenchmarkSweep(search_space={"top_k": {"values": [2]}}, top_n_llm=1)
        with pytest.raises(RuntimeError, match="judge failed"):
            await grid_engine.run_sweep(sweep, judge_model="judge")
        assert _settings_overrides.get() == {"cache_enabled": True, "retriever_top_k": 99}
    finally:
        _settings_overrides.reset(token)


@pytest.mark.asyncio
async def test_chunk_corpus_stream_closes_result_on_early_stop():
    class Rows:
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            return ("content", "internal_private", 10, None)

        async def close(self):
            self.closed = True

    rows = Rows()
    session = SimpleNamespace(stream=AsyncMock(return_value=rows))
    stream = SQLAlchemyChunkRepository(session).iter_all_contents_with_acl(batch_size=7)
    assert await anext(stream) == ("content", "internal_private", 10, None)
    await stream.aclose()
    assert rows.closed
    assert session.stream.call_args.args[0].get_execution_options()["yield_per"] == 7


@pytest.mark.asyncio
async def test_chunk_bulk_replacement_keeps_acl_and_metadata_in_shared_session():
    added = []

    async def flush():
        for idx, item in enumerate(added, 10):
            item.id = idx

    session = SimpleNamespace(
        execute=AsyncMock(), add_all=lambda items: added.extend(items), flush=AsyncMock(side_effect=flush)
    )
    repository = SQLAlchemyChunkRepository(session)
    ids = await repository.bulk_insert(
        7,
        "report.pdf",
        "internal_group",
        ["first", "second"],
        group_id=3,
        content_hashes=["hash1", "hash2"],
        sections=["A", "B"],
        act_version_id=5,
    )
    assert ids == [10, 11]
    assert [(m.document_id, m.group_id, m.visibility, m.act_version_id) for m in added] == [
        (7, 3, "internal_group", 5),
        (7, 3, "internal_group", 5),
    ]
    assert [(m.chunk_index, m.content_hash, m.section) for m in added] == [
        (0, "hash1", "A"),
        (1, "hash2", "B"),
    ]
    assert session.execute.call_args.args[0].compile().params == {"document_id_1": 7}
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_chunk_replacement_still_removes_old_chunks():
    session = SimpleNamespace(execute=AsyncMock(), add_all=MagicMock(), flush=AsyncMock())
    assert await SQLAlchemyChunkRepository(session).bulk_insert(7, "report.pdf", "internal_public", []) == []
    session.execute.assert_awaited_once()
    session.add_all.assert_not_called()
    session.flush.assert_not_awaited()
