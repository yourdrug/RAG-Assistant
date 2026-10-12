"""Regression tests for load, invalidation and incremental BM25 updates."""

import asyncio
import threading
from unittest.mock import Mock

import pytest

from domain.value_objects.visibility import DocumentVisibility
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.bm25_updater import bm25_add, bm25_replace, bm25_remove
from infrastructure.ml.clients import factories
from infrastructure.ml.clients.client_registry import MLClientRegistry


def test_invalidation_reloads_immediately(monkeypatch):
    loader = Mock(side_effect=[BM25Index(["old"]), BM25Index(["fresh"])])
    monkeypatch.setattr(factories, "load_bm25_index", loader)
    registry = MLClientRegistry()
    registry.bm25_index()
    registry.invalidate_bm25()
    assert registry.bm25_index().texts == ["fresh"]


@pytest.mark.asyncio
async def test_invalidation_discards_inflight_snapshot(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release loader")
            return BM25Index(["stale"])
        return BM25Index(["fresh"])

    monkeypatch.setattr(factories, "load_bm25_index", load)
    registry = MLClientRegistry()
    pending = asyncio.create_task(registry.ensure_bm25_loaded())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        registry.invalidate_bm25()
    finally:
        release.set()
    index = await asyncio.wait_for(pending, 5)
    assert index.texts == ["fresh"]
    assert registry.bm25_index() is index


@pytest.mark.asyncio
async def test_sync_update_and_async_load_share_one_snapshot(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(5):
            raise TimeoutError("test did not release loader")
        return BM25Index(["original"], hashes=["original-hash"])

    monkeypatch.setattr(factories, "load_bm25_index", load)
    registry = MLClientRegistry()
    pending = asyncio.create_task(registry.ensure_bm25_loaded())
    update = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        update = asyncio.create_task(asyncio.to_thread(bm25_add, registry, "new", "new-hash"))
        await asyncio.sleep(0)
    finally:
        release.set()
    await asyncio.wait_for(asyncio.gather(pending, update), 5)
    assert registry.bm25_index().hashes == ["original-hash", "new-hash"]
    assert calls == 1


@pytest.mark.parametrize("operation", ["add", "replace", "remove"])
def test_updates_after_invalidation_are_applied(monkeypatch, operation):
    monkeypatch.setattr(factories, "load_bm25_index", lambda: BM25Index(["old"], hashes=["old-hash"]))
    registry = MLClientRegistry()
    registry.bm25_index()
    registry.invalidate_bm25()
    if operation == "add":
        bm25_add(registry, "new", "new-hash")
        expected = ["old-hash", "new-hash"]
    elif operation == "replace":
        bm25_replace(registry, "old-hash", "new", "new-hash")
        expected = ["new-hash"]
    else:
        bm25_remove(registry, "old-hash")
        expected = []
    assert registry.bm25_index().hashes == expected


def test_add_without_snapshot_retains_text_and_acl(monkeypatch):
    monkeypatch.setattr(factories, "load_bm25_index", lambda: None)
    registry = MLClientRegistry()
    bm25_add(registry, "new", "new-hash", DocumentVisibility.INTERNAL_PRIVATE.value, 42, None)
    index = registry.bm25_index()
    assert index is not None
    assert index.hashes == ["new-hash"]
    assert index.doc_visibility == [DocumentVisibility.INTERNAL_PRIVATE.value]
    assert index.doc_owner_id == [42]


def test_invalidation_between_selection_and_mutation_retries(monkeypatch):
    monkeypatch.setattr(factories, "load_bm25_index", lambda: BM25Index(["old"], hashes=["old-hash"]))
    registry = MLClientRegistry()
    original_accessor = registry.bm25_index
    invalidate = True

    def select_then_invalidate():
        nonlocal invalidate
        index = original_accessor()
        if invalidate:
            invalidate = False
            registry.invalidate_bm25()
        return index

    monkeypatch.setattr(registry, "bm25_index", select_then_invalidate)
    bm25_add(registry, "new", "new-hash")
    assert original_accessor().hashes == ["old-hash", "new-hash"]


def test_failed_load_can_retry(monkeypatch):
    loader = Mock(side_effect=[OSError("storage unavailable"), BM25Index(["recovered"])])
    monkeypatch.setattr(factories, "load_bm25_index", loader)
    registry = MLClientRegistry()
    with pytest.raises(OSError, match="storage unavailable"):
        registry.bm25_index()
    assert registry.bm25_index().texts == ["recovered"]
