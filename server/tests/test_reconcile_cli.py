"""Reconciliation uses the bootstrapped corpus and preserves existing SQL documents."""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import QdrantClient, models
from typer.testing import CliRunner

from composition.cli.cli import cli
from composition.cli.commands import reconcile
from domain.shared.exceptions import ValidationError
from infrastructure.corpus.locks import validate_write_admission
from infrastructure.corpus.repository import CorpusPhase
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.database.database import database
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.repositories.vector.qdrant_vector_store_repository import QdrantVectorStoreRepository
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from infrastructure.corpus.repository import CorpusRow


@pytest.fixture
def reconcile_corpus() -> tuple[QdrantClient, CorpusSnapshot]:
    raw = QdrantClient(":memory:")
    snapshot = CorpusSnapshot("http://qdrant:6333", "alias", "physical", "identity", "generation")

    for name in ("physical", "next"):
        raw.create_collection(
            name, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        raw.upsert(
            name,
            [
                models.PointStruct(
                    id=doc_id,
                    vector=[1.0, 0.0],
                    payload={"metadata": {"document_id": doc_id}},
                )
                for doc_id in (10, 20, 30)
            ],
        )

    return raw, snapshot


@pytest.fixture
def reconciliation(
    monkeypatch: pytest.MonkeyPatch, reconcile_corpus: tuple[QdrantClient, CorpusSnapshot]
) -> SimpleNamespace:
    raw, snapshot = reconcile_corpus
    active = snapshot
    env = SimpleNamespace(
        generation_changes=False, admitted=[], remaining={}, lifecycle=[], snapshot=snapshot
    )

    @contextmanager
    def guard(selected: CorpusSnapshot) -> Iterator[None]:
        validate_write_admission(
            cast("CorpusRow", {"phase": CorpusPhase.ACTIVE, "active": asdict(active)}), selected
        )
        env.admitted.append(selected)
        yield

    registry = MLClientRegistry()
    factory = FakeUnitOfWorkFactory()
    infra = SimpleNamespace(
        ml=SimpleNamespace(clients=registry, vector_index_writer=QdrantVectorStoreRepository(registry)),
        db=SimpleNamespace(uow=factory),
    )

    async def postgres_ids() -> tuple[set[int], set[int]]:
        nonlocal active

        if env.generation_changes:
            active = replace(snapshot, physical_collection="next", generation="new-generation")
            # Simulate a journal/alias switch while the scan still owns its lease.
            registry.install_corpus(BoundQdrantClient(raw, active, guard))

        # Document 20 exists with status FAILED. Its points must be preserved.
        return {10, 20}, {10}

    async def run(action: Callable[[object], Awaitable[int]]) -> int:
        env.lifecycle.append("bootstrap")
        registry.install_corpus(BoundQdrantClient(raw, snapshot, guard))
        try:
            return await action(infra)
        finally:
            for name in ("physical", "next"):
                env.remaining[name] = {point.id for point in raw.scroll(name)[0]}

            env.lifecycle.append("dispose")
            await registry.close()

    monkeypatch.setattr(reconcile, "run_with_infrastructure", run)

    async def chunk_ids() -> set[int]:
        return set(env.chunk_ids)

    env.chunk_ids = {10, 20}
    monkeypatch.setattr(reconcile, "get_all_postgres_doc_ids", postgres_ids)
    monkeypatch.setattr(reconcile, "get_all_postgres_chunk_ids", chunk_ids)
    return env


@pytest.mark.parametrize("delete", [False, True])
@pytest.mark.parametrize("generation_changes", [False, True])
def test_reconcile_pins_generation_and_deletes_only_true_orphans(
    reconciliation: SimpleNamespace, *, delete: bool, generation_changes: bool
) -> None:
    env = reconciliation
    env.generation_changes = generation_changes
    result = CliRunner().invoke(cli.cli, ["reconcile", "run", *(["--delete"] if delete else [])])
    assert result.exit_code == 1  # Drift was detected, including after successful repair.
    assert env.lifecycle == ["bootstrap", "dispose"]
    assert env.remaining["next"] == {10, 20, 30}

    if delete and generation_changes:
        assert isinstance(result.exception, ValidationError)
        assert "generation changed" in str(result.exception)
    else:
        assert isinstance(result.exception, SystemExit)

    if delete and not generation_changes:
        assert env.remaining["physical"] == {10, 20}
        assert env.admitted == [env.snapshot]
    else:
        assert env.remaining["physical"] == {10, 20, 30}
        assert env.admitted == []


@pytest.mark.asyncio
async def test_reconcile_reads_primary_without_owning_database_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = MagicMock()
    result.fetchall.return_value = [(10, "done"), (20, "failed")]
    session = SimpleNamespace(execute=AsyncMock(return_value=result), close=AsyncMock())
    primary = MagicMock(return_value=session)
    monkeypatch.setattr(database, "get_write_session", primary)
    monkeypatch.setattr(database, "get_read_session", MagicMock(side_effect=AssertionError("replica")))
    assert await reconcile.get_all_postgres_doc_ids() == ({10, 20}, {10})
    primary.assert_called_once_with()
    session.close.assert_awaited_once()


@pytest.mark.parametrize("delete", [False, True])
def test_reconcile_removes_stale_chunk_points_of_live_documents(
    reconciliation: SimpleNamespace, reconcile_corpus: tuple[QdrantClient, CorpusSnapshot], *, delete: bool
) -> None:
    """Re-chunking replaces chunk ids; a point whose chunk row is gone is stale."""
    raw, _snapshot = reconcile_corpus
    raw.upsert(
        "physical",
        [
            models.PointStruct(id=point_id, vector=[1.0, 0.0], payload={"metadata": {"document_id": 10}})
            for point_id in (11, 12)
        ],
    )
    env = reconciliation
    env.chunk_ids = {10, 11, 20}  # Chunk 12 of live document 10 was replaced.

    result = CliRunner().invoke(cli.cli, ["reconcile", "run", *(["--delete"] if delete else [])])

    assert result.exit_code == 1
    expected = {10, 11, 20} if delete else {10, 11, 12, 20, 30}
    assert env.remaining["physical"] == expected


def test_stale_points_skip_unknown_documents_and_current_chunks() -> None:
    points = {1: 7, 2: 7, 3: 8, 4: 9}
    assert reconcile.find_stale_chunk_points(points, postgres_ids={7, 8}, chunk_ids={1, 3}) == {7: [2]}
