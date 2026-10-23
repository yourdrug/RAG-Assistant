"""Typed transitions and corpus ownership across application RAG exits."""

from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document
import pytest

from application.rag.dto import (
    CompletedAnswer,
    ContextBudget,
    PreparedContext,
    PreparedQuery,
    RagRequest,
    RetrievalResult,
    SanitizedAnswer,
    Sufficiency,
    TerminalResponse,
)
from application.rag.service import RagService
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.stream_events import TextChunk
from infrastructure.ml.rag.langchain_mapping import lc_documents, scored_values
from infrastructure.ml.rag.rag_config import build_rag_settings


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def workflow() -> tuple[RagService, MagicMock, list[str]]:
    request = RagRequest(
        "question", (), RagContext(1, "internal", "user"), build_rag_settings(), 0, "req", "model"
    )
    query = PreparedQuery(request, "question")
    retrieval = RetrievalResult(query, Breadth.NARROW, "general", 25, 4, 1, 1, False, (), (), (), False, 0)
    context = PreparedContext(retrieval, (), (), (), ContextBudget(4096, 400, 0, 0), None, None)
    answer = SanitizedAnswer(context, "safe answer", None, 0.1)
    operations = MagicMock()
    operations.request.return_value = request
    operations.condense = AsyncMock(return_value=query)
    operations.scan = AsyncMock(return_value=None)
    operations.lookup = AsyncMock(return_value=query)
    operations.retrieve = AsyncMock(return_value=retrieval)
    operations.reject_ood = AsyncMock(return_value=None)
    operations.relevant = AsyncMock(return_value=True)
    operations.assess = AsyncMock(return_value=Sufficiency(False, "refined"))
    operations.reject = AsyncMock(return_value=TerminalResponse((TextChunk("rejected"),)))
    operations.build.return_value = context
    operations.generate = AsyncMock(return_value=answer)
    operations.finalize.return_value = CompletedAnswer(answer, (), 0)
    operations.store = AsyncMock()
    leases = []

    @asynccontextmanager
    async def corpus() -> AsyncIterator[None]:
        leases.append("acquired")
        try:
            yield
        finally:
            leases.append("released")

    return (
        RagService(
            preparation=operations,
            safety=operations,
            cache=operations,
            retrieval=operations,
            evidence=operations,
            context=operations,
            generation=operations,
            finalization=operations,
            corpus=corpus,
        ),
        operations,
        leases,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_at", ["input", "cache", "relevance", "success", "error", "close"])
async def test_corpus_lease_covers_all_stream_exits(exit_at: str) -> None:
    service, operations, leases = workflow()
    terminal = TerminalResponse((TextChunk("terminal"),))

    if exit_at == "input":
        operations.scan.return_value = terminal
    elif exit_at == "cache":
        operations.lookup.return_value = terminal
    elif exit_at == "relevance":
        operations.relevant.return_value = False
    elif exit_at == "error":
        operations.generate.side_effect = RuntimeError("failed")

    stream = service.stream("question", [], RagContext(1, "internal"))

    if exit_at == "close":
        await anext(stream)
        assert leases == ["acquired"]
        await stream.aclose()
    elif exit_at == "error":
        with pytest.raises(RuntimeError, match="failed"):
            _ = [event async for event in stream]
    else:
        _ = [event async for event in stream]

    assert leases == ["acquired", "released"]

    if exit_at == "relevance":
        assert operations.retrieve.await_count == 2
        operations.assess.assert_awaited_once()
        retried = operations.retrieve.await_args.args[0]
        assert retried.query == "refined"
        assert retried.request is operations.request.return_value


@pytest.mark.asyncio
async def test_cancellation_releases_corpus_while_generation_is_waiting() -> None:
    service, operations, leases = workflow()
    generating = asyncio.Event()

    async def generate(context: PreparedContext) -> SanitizedAnswer:
        generating.set()
        await asyncio.Event().wait()
        return SanitizedAnswer(context, "", None, None)

    operations.generate.side_effect = generate

    async def consume() -> None:
        async for _ in service.stream("question", [], RagContext(1, "internal")):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(generating.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert leases == ["acquired", "released"]
    operations.store.assert_not_awaited()


def test_new_application_modules_do_not_import_adapters() -> None:
    app = Path(__file__).parents[1] / "app" / "application"
    files = list((app / "rag").glob("*.py")) + list((app / "benchmark").glob("sweep*.py"))
    forbidden = ("infrastructure", "config", "langchain", "qdrant")

    for file in files:
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden), file
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith(forbidden) for alias in node.names), file


def test_mypy_rejects_generation_and_finalization_before_required_stages(tmp_path: Path) -> None:
    sample = tmp_path / "invalid_rag_transitions.py"
    sample.write_text(
        "from application.rag.dto import PreparedQuery, RetrievalResult\n"
        "from application.rag.ports import Generation, Finalization\n"
        "async def invalid(g: Generation, f: Finalization, q: PreparedQuery, r: RetrievalResult) -> None:\n"
        "    await g.generate(q)\n"
        "    f.finalize(r)\n"
    )
    result = subprocess.run(  # noqa: S603 - invokes the local type checker on a fixed test sample.
        [sys.executable, "-m", "mypy", "--follow-imports=silent", str(sample)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.count("[arg-type]") == 2, result.stdout
    assert 'expected "PreparedContext"' in result.stdout
    assert 'expected "SanitizedAnswer"' in result.stdout


@pytest.mark.asyncio
async def test_invoke_collects_stream_values_using_one_request_snapshot() -> None:
    service, operations, leases = workflow()
    result = await service.invoke("question", [], RagContext(1, "internal"))
    assert result.answer == "safe answer"
    assert result.sources == []
    assert result.confidence == 0
    assert result.breadth == Breadth.NARROW
    assert result.domain == "general"
    assert result.ttft_sec == 0.1
    assert result.model_used == "model"
    operations.request.assert_called_once()
    assert leases == ["acquired", "released"]


def test_document_stage_conversion_preserves_acl_temporal_and_structural_metadata() -> None:
    metadata = {
        "document_id": 42,
        "visibility": "internal_group",
        "owner_id": 7,
        "group_id": 3,
        "act_version_id": 9,
        "effective_from": "2025-01-01",
        "effective_to": "2026-01-01",
        "table_id": "table-1",
        "table_row_start": 2,
        "citation_id": 4,
        "domain_metadata": {"act_number": "17"},
    }
    doc = Document(page_content="evidence", metadata=metadata)
    values = scored_values([(doc, 0.75)])
    converted = lc_documents(values)
    assert converted[0][0].page_content == "evidence"
    assert converted[0][0].metadata == metadata
    assert converted[0][1] == 0.75
    converted[0][0].metadata["domain_metadata"]["act_number"] = "changed"
    assert values[0].chunk.metadata["domain_metadata"]["act_number"] == "17"
    assert doc.metadata["domain_metadata"]["act_number"] == "17"
