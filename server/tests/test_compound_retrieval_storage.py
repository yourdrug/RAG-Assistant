"""Standalone retrieval experiments keep their sparse snapshot and storage owner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from infrastructure.ml.clients import factories


if TYPE_CHECKING:
    from types import ModuleType

    from langchain_core.documents import Document

    from infrastructure.budget.runtime import BudgetRuntime


@pytest.fixture
def experiment() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "benchmark/compound_retrieval_experiment.py"
    spec = importlib.util.spec_from_file_location("compound_retrieval_experiment", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_search", [False, True])
async def test_live_experiment_uses_runtime_sparse_search_without_local_storage(
    experiment: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail_search: bool
) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["existing sparse evidence"],"hashes":["sparse-hash"]}')
    storage = SimpleNamespace(download_to_temp=AsyncMock(return_value=path), aclose=AsyncMock())
    monkeypatch.setattr("infrastructure.storage.file_storage.create_storage", lambda *_: storage)

    def create_reranker(*, runtime: BudgetRuntime | None = None) -> SimpleNamespace:
        assert runtime is None
        return SimpleNamespace(close=AsyncMock())

    monkeypatch.setattr(factories, "create_reranker", create_reranker)
    snapshots = []

    async def search(
        query: str, fetch_k: int, access_filter: object, rag: object, registry: object, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        snapshots.append(registry)
        assert not hasattr(registry, "ensure_bm25_loaded")

        if fail_search:
            msg = "controlled retrieval failure"
            raise RuntimeError(msg)

        return []

    monkeypatch.setattr(experiment, "run_hybrid_search", search)
    monkeypatch.setattr(experiment, "rerank_documents", AsyncMock(return_value=[]))
    cases = [{"id": 149, "question": "question"}]

    if fail_search:
        with pytest.raises(RuntimeError, match="controlled retrieval failure"):
            await experiment.compare(cases, 1)
    else:
        await experiment.compare(cases, 1)

    assert snapshots
    assert all(index is snapshots[0] for index in snapshots)
    storage.download_to_temp.assert_not_awaited()
    storage.aclose.assert_not_awaited()
    assert path.exists()
