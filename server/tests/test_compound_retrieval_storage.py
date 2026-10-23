"""Standalone retrieval experiments keep their sparse snapshot and storage owner."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from infrastructure.ml.clients import factories


@pytest.fixture
def experiment():
    path = Path(__file__).parents[2] / "scripts" / "benchmark/compound_retrieval_experiment.py"
    spec = importlib.util.spec_from_file_location("compound_retrieval_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_search", [False, True])
async def test_live_experiment_loads_sparse_and_closes_owned_storage(
    experiment, monkeypatch, tmp_path, fail_search
):
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["existing sparse evidence"],"hashes":["sparse-hash"]}')
    storage = SimpleNamespace(download_to_temp=AsyncMock(return_value=path), aclose=AsyncMock())
    monkeypatch.setattr("infrastructure.storage.file_storage.create_storage", lambda: storage)

    def create_reranker(*, runtime=None):
        assert runtime is None
        return SimpleNamespace(close=AsyncMock())

    monkeypatch.setattr(factories, "create_reranker", create_reranker)
    snapshots = []

    async def search(query, fetch_k, access_filter, rag, registry, **kwargs):
        snapshots.append(await registry.ensure_bm25_loaded())
        if fail_search:
            raise RuntimeError("controlled retrieval failure")
        return []

    monkeypatch.setattr(experiment, "run_hybrid_search", search)
    monkeypatch.setattr(experiment, "rerank_documents", AsyncMock(return_value=[]))
    cases = [{"id": 149, "question": "question"}]
    if fail_search:
        with pytest.raises(RuntimeError, match="controlled retrieval failure"):
            await experiment.compare(cases, 1)
    else:
        await experiment.compare(cases, 1)
    assert snapshots and all(index is not None and index.hashes == ["sparse-hash"] for index in snapshots)
    assert all(index is snapshots[0] for index in snapshots)
    storage.download_to_temp.assert_awaited_once()
    storage.aclose.assert_awaited_once()
    assert not path.exists()
