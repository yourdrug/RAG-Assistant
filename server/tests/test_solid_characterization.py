"""Public workflow contracts captured before splitting the implementation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast
from unittest.mock import AsyncMock, MagicMock

from langchain.schema import Document
import pytest

from application.benchmark.sweep_ports import SweepCancelled
from composition.service_providers import build_sweep_engine
from config import _settings_overrides, get_setting, settings
from domain.benchmark.sweep import BenchmarkSweep
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.repositories.chunk import ChunkInsertOptions
from infrastructure.benchmark.evaluation import metrics
from infrastructure.benchmark.evaluation.runner import run_benchmark_async
from infrastructure.benchmark.judging import context_judge, evidence_judge, judge
from infrastructure.benchmark.reporting import progress as persistence
from infrastructure.ml.rag.benchmark_evidence import capture_prompt
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from application.benchmark.sweep import SweepEngine


@pytest.fixture
def benchmark_io(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], AsyncMock]:
    # These characterize the individual judge path mocked below.
    monkeypatch.setattr(settings, "benchmark_judge_grouped_enabled", False)
    questions = [{"id": "first", "question": "question", "source_hint": "report"}]
    monkeypatch.setattr(
        evidence_judge, "judge_evidence", lambda *_args, **_kwargs: {"scores": {}, "details": {}}
    )
    score = AsyncMock(return_value={"faithfulness": 8, "relevancy": 7, "correctness": None})
    monkeypatch.setattr(judge, "judge_answer_async", score)
    monkeypatch.setattr(
        context_judge, "compute_context_precision_recall", lambda *_a, **_kw: {"context_precision": 6}
    )
    monkeypatch.setattr(metrics, "estimate_cost_usd", lambda *_a: 0.25)
    monkeypatch.setattr(persistence, "log_question_result", lambda *_a: None)
    monkeypatch.setattr(persistence, "log_summary", lambda *_a: None)
    return questions, score


@pytest.mark.asyncio
async def test_full_benchmark_preserves_results_repeated_runs_and_cache_scope(
    benchmark_io: tuple[list[dict[str, Any]], AsyncMock],
) -> None:
    questions, score = benchmark_io
    contexts = []

    async def invoke(question: object, history: object, ctx: Any) -> SimpleNamespace:

        assert get_setting("cache_enabled") is False
        contexts.append(ctx)
        docs = [(Document(page_content="context", metadata={"source": "report.pdf"}), 0.9)]
        capture_prompt(docs, docs, "context")
        return SimpleNamespace(
            answer="answer",
            input_tokens=20,
            output_tokens=10,
            generation_sec=0.5,
            ttft_sec=0.1,
            breadth="narrow",
            domain="general",
            sources=[{"source": "report.pdf", "max_score": 0.9, "content": "context"}],
        )

    previous = _settings_overrides.get()
    results = await run_benchmark_async(
        questions, 4, "judge", n_runs=2, rag_service=SimpleNamespace(invoke=invoke)
    )
    assert _settings_overrides.get() == previous
    assert [r["run"] for r in results] == [1, 2]
    assert [r["id"] for r in results] == ["first", "first"]
    assert results[0]["retriever_metrics"]["hit_rate"] == 1
    assert results[0]["estimated_generation_cost_usd"] == 0.25
    assert results[0]["cost_usd"] == 0  # The test double makes no model calls.
    assert results[0]["input_tokens"] == 20
    assert all(c.user_role == "admin" and c.user_kind == "internal" for c in contexts)
    assert score.call_args.kwargs["context"] == "context"


@pytest.mark.asyncio
async def test_benchmark_restores_cache_scope_and_propagates_pipeline_error(
    benchmark_io: tuple[list[dict[str, Any]], AsyncMock],
) -> None:
    questions, _ = benchmark_io
    previous = _settings_overrides.get()
    rag = SimpleNamespace(invoke=AsyncMock(side_effect=RuntimeError("pipeline failed")))
    with pytest.raises(RuntimeError, match="pipeline failed"):
        await run_benchmark_async(questions, 4, "judge", rag_service=rag)
    assert _settings_overrides.get() == previous


@pytest.fixture
def grid_engine(monkeypatch: pytest.MonkeyPatch) -> SweepEngine:
    engine = build_sweep_engine(uow_factory=FakeUnitOfWorkFactory())
    monkeypatch.setattr(
        engine, "load_questions", AsyncMock(return_value=[{"question": "q", "source_hint": "doc"}])
    )
    monkeypatch.setattr(engine, "cache_candidates", AsyncMock(return_value=({}, {}, {})))
    monkeypatch.setattr(
        engine, "score_config_cheap", lambda cfg, *_a, **_kwargs: {"composite_score": cfg["top_k"]}
    )
    return engine


@pytest.mark.asyncio
async def test_grid_sweep_scores_sorts_and_reports_progress(grid_engine: Any) -> None:
    progress = []
    sweep = BenchmarkSweep(search_space={"top_k": {"values": [2, 5]}}, top_n_llm=0)
    results = await grid_engine.run_sweep(
        sweep, progress_callback=lambda *args, **_kwargs: progress.append(args)
    )
    assert [r["config"]["top_k"] for r in results] == [5, 2]
    assert [(p[0], p[1]) for p in progress] == [(1, 2), (2, 2)]


@pytest.mark.asyncio
async def test_sweep_cancels_before_scoring(grid_engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    score = MagicMock()
    monkeypatch.setattr(grid_engine, "score_config_cheap", score)
    sweep = BenchmarkSweep(search_space={"top_k": {"values": [2]}}, top_n_llm=0)
    with pytest.raises(SweepCancelled):
        await grid_engine.run_sweep(sweep, should_cancel=AsyncMock(return_value=True))
    score.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_restores_existing_overrides_when_full_benchmark_fails(
    grid_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
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
async def test_chunk_corpus_stream_closes_result_on_early_stop() -> None:
    class Rows:
        closed = False

        def __aiter__(self) -> Self:
            return self

        async def __anext__(self) -> tuple[str, str, int, None]:
            return ("content", "internal_private", 10, None)

        async def close(self) -> None:
            self.closed = True

    rows = Rows()
    session = SimpleNamespace(stream=AsyncMock(return_value=rows))
    stream = SQLAlchemyChunkRepository(cast("AsyncSession", session)).iter_all_contents_with_acl(batch_size=7)
    assert await anext(stream) == ("content", "internal_private", 10, None)
    await cast("AsyncGenerator[object, None]", stream).aclose()
    assert rows.closed
    assert session.stream.call_args.args[0].get_execution_options()["yield_per"] == 7


@pytest.mark.asyncio
async def test_chunk_bulk_replacement_keeps_acl_and_metadata_in_shared_session() -> None:
    added: list[Any] = []

    async def flush() -> None:
        for idx, item in enumerate(added, 10):
            item.id = idx

    session = SimpleNamespace(execute=AsyncMock(), add_all=added.extend, flush=AsyncMock(side_effect=flush))
    repository = SQLAlchemyChunkRepository(cast("AsyncSession", session))
    ids = await repository.bulk_insert(
        7,
        "report.pdf",
        "internal_group",
        [
            ChunkInsert(content="first", content_hash="hash1", section="A"),
            ChunkInsert(content="second", content_hash="hash2", section="B"),
        ],
        options=ChunkInsertOptions(group_id=3, act_version_id=5),
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
async def test_empty_chunk_replacement_still_removes_old_chunks() -> None:
    session = SimpleNamespace(execute=AsyncMock(), add_all=MagicMock(), flush=AsyncMock())
    assert (
        await SQLAlchemyChunkRepository(cast("AsyncSession", session)).bulk_insert(
            7, "report.pdf", "internal_public", []
        )
        == []
    )
    session.execute.assert_awaited_once()
    session.add_all.assert_not_called()
    session.flush.assert_not_awaited()
