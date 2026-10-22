from unittest.mock import AsyncMock, Mock
from dataclasses import asdict
from time import monotonic
from infrastructure.corpus.control import PostgresCorpusControl
import pytest
from infrastructure.corpus.snapshot import CorpusSnapshot, BoundQdrantClient
from qdrant_client import QdrantClient, models
from config import settings
from infrastructure.corpus.clients import probe_active_corpus
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, Modifier
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME, point_vectors
import hashlib
from dataclasses import replace
from domain.shared.exceptions import ValidationError
from infrastructure.corpus.validation import validate_corpus
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.cache.answers import _cache_key


def test_bound_client_never_mixes_new_setting_with_old_collection():
    raw = Mock()
    snapshot = CorpusSnapshot('http://qdrant:6333', 'docs', 'docs_v1', 'identity')
    client = BoundQdrantClient(raw, snapshot)
    client.search(collection_name='docs_v2', query_vector=[1], query_filter='acl')
    assert raw.search.call_args.kwargs['collection_name'] == 'docs_v1'
    assert raw.search.call_args.kwargs['query_filter'] == 'acl'
    client.scroll(collection_name='docs_v2', scroll_filter='acl')
    assert raw.scroll.call_args.kwargs['collection_name'] == 'docs_v1'
    assert raw.scroll.call_args.kwargs['scroll_filter'] == 'acl'


def test_binding_collection_does_not_get_rewritten():
    raw = Mock()
    client = BoundQdrantClient(raw, CorpusSnapshot('http://qdrant:6333', 'docs', 'docs_v1', 'i'))
    client.get_collection('docs_v1__embedding_identity')
    raw.get_collection.assert_called_once_with('docs_v1__embedding_identity')


def fixture_corpus():
    client = QdrantClient(':memory:')
    identity = 'tei:model:revision'
    client.create_collection(
        'docs_v1', vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )
    client.create_collection(
        'docs_v1__embedding_identity',
        vectors_config={
            hashlib.sha256(identity.encode()).hexdigest(): VectorParams(size=1, distance=Distance.COSINE)
        },
    )
    row = {
        'id': 1,
        'document_id': 10,
        'visibility': 'internal_private',
        'owner_id': 7,
        'group_id': None,
        'content_hash': 'abc',
        'doc_domain': 'general',
        'content': 'private text',
    }
    client.upsert(
        'docs_v1',
        [
            PointStruct(
                id=1,
                vector=point_vectors(row['content'], [1.0, 0.0]),
                payload={
                    'page_content': row['content'],
                    'metadata': {
                        **{k: v for k, v in row.items() if k not in {'id', 'content'}},
                        'embedding_model': identity,
                    },
                },
            )
        ],
    )
    return client, identity, row


def test_candidate_requires_exact_pg_acl_and_immutable_identity():
    client, identity, row = fixture_corpus()
    try:
        snapshot = validate_corpus(client, 'http://qdrant:6333', 'docs_v1', identity, 2, [row])
        assert snapshot.physical_collection == 'docs_v1'
        for bad_row in [
            {**row, 'owner_id': 8},
            {**row, 'document_id': 11},
            {**row, 'content': 'different'},
            {**row, 'visibility': 'internal_public'},
        ]:
            with pytest.raises(ValidationError):
                validate_corpus(client, snapshot.endpoint, 'docs_v1', identity, 2, [bad_row])
        with pytest.raises(ValidationError):
            validate_corpus(client, snapshot.endpoint, 'docs_v1', 'other:model', 2, [row])
        with pytest.raises(ValidationError):
            validate_corpus(client, snapshot.endpoint, 'docs_v1', identity, 3, [row])
        with pytest.raises(ValidationError):
            validate_corpus(client, snapshot.endpoint, 'docs_v1', identity, 2, [])
    finally:
        client.close()


@pytest.mark.asyncio
async def test_request_keeps_old_client_bm25_and_cache_namespace_until_finished():
    registry = MLClientRegistry()
    raw_old, raw_new = Mock(), Mock()
    old_snapshot = CorpusSnapshot('http://qdrant:6333', 'docs', 'v1', 'i', 'generation1')
    new_snapshot = replace(old_snapshot, physical_collection='v2', generation='generation2')
    old = BoundQdrantClient(raw_old, old_snapshot)
    new = BoundQdrantClient(raw_new, new_snapshot)
    old_index, new_index = object(), object()
    registry.install_corpus(old, old_index)
    async with registry.corpus_operation():
        old_key = _cache_key('question', 'scope')
        registry.install_corpus(new, new_index)
        assert registry.qdrant_client() is old
        assert await registry.ensure_bm25_loaded() is old_index
        assert _cache_key('question', 'scope') == old_key
        raw_old.close.assert_not_called()
    await old.aclose()
    raw_old.close.assert_called_once()
    async with registry.corpus_operation():
        assert registry.qdrant_client() is new
        assert await registry.ensure_bm25_loaded() is new_index
        assert _cache_key('question', 'scope') != old_key
    await registry.close()


@pytest.mark.asyncio
async def test_corpus_admission_does_not_load_legacy_bm25():
    from unittest.mock import AsyncMock

    loader = AsyncMock(side_effect=AssertionError("S3 snapshot must not be loaded"))
    registry = MLClientRegistry(bm25_loader=loader)
    snapshot = CorpusSnapshot('http://qdrant:6333', 'docs', 'v1', 'i', 'g1')
    registry.install_corpus(BoundQdrantClient(Mock(), snapshot), None)
    registry.invalidate_bm25()
    try:
        async with registry.corpus_operation():
            assert registry.qdrant_client().snapshot == snapshot
        loader.assert_not_awaited()
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_generic_config_update_cannot_bypass_corpus_switch():
    from application.admin import ConfigService
    from fakes import FakeUnitOfWorkFactory

    service = ConfigService(FakeUnitOfWorkFactory(), Mock())
    for key in ['qdrant_url', 'collection_name']:
        with pytest.raises(ValidationError, match='coordinated'):
            await service.update_parameter(key, 'anything')


def test_candidate_origin_requires_deployment_approval():
    from infrastructure.corpus.clients import validate_corpus_endpoint

    validate_corpus_endpoint('http://qdrant:6333', 'http://qdrant:6333')
    with pytest.raises(ValidationError):
        validate_corpus_endpoint('https://attacker.invalid', 'http://qdrant:6333')


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['active', 'failed', 'preparing', 'committing'])
@pytest.mark.parametrize('already_prepared', [False, True])
async def test_poll_catches_up_to_durable_active_generation(monkeypatch, phase, already_prepared):
    from dataclasses import asdict
    from unittest.mock import AsyncMock
    from infrastructure.corpus import control as module
    from config import settings

    old = CorpusSnapshot('http://qdrant:6333', 'docs', 'v1', 'i', 'old')
    active = replace(old, physical_collection='v2', generation='new')
    candidate = replace(active, physical_collection='v3', generation='pending')
    registry = MLClientRegistry()
    registry.install_corpus(BoundQdrantClient(Mock(), old), object())
    listener = Mock(synchronize=AsyncMock())
    control = module.PostgresCorpusControl(Mock(), registry, listener)
    control.read_state = AsyncMock(
        return_value={
            'phase': phase,
            'active': asdict(active),
            'candidate': asdict(active if phase == 'committing' else candidate)
            if phase in {'preparing', 'committing'}
            else None,
            'digest': 'stale',
        }
    )
    if already_prepared:
        control.prepared = BoundQdrantClient(Mock(), candidate)
        control.prepared_index = object()
    control.chunks = AsyncMock(return_value=[])
    control.heartbeat = AsyncMock()
    control.recover = AsyncMock()
    monkeypatch.setattr(module, 'raw_corpus_client', Mock(return_value=Mock()))
    monkeypatch.setattr(module.corpus_snapshot, 'installed_corpus', None)
    monkeypatch.setattr(settings, 'qdrant_url', old.endpoint)
    monkeypatch.setattr(settings, 'collection_name', old.alias)
    control.prepare = AsyncMock(return_value=(BoundQdrantClient(Mock(), candidate), object()))
    try:
        await control.tick()
        assert registry.corpus_snapshot == active
        assert await registry.ensure_bm25_loaded() is None
        listener.synchronize.assert_awaited_once()
        assert control.error is None
        await control.tick()
        listener.synchronize.assert_awaited_once()
    finally:
        if control.prepared is not None:
            control.prepared.close()
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['none', 'generation', 'monitor', 'stalled', 'error', 'database'])
async def test_corpus_readiness_fences_stale_replicas(fault):
    registry = MLClientRegistry()
    snapshot = CorpusSnapshot('http://qdrant:6333', 'docs', 'v1', 'i', 'old')
    registry.install_corpus(BoundQdrantClient(Mock(), snapshot), object())
    control = PostgresCorpusControl(Mock(), registry, None)
    control.task = Mock(done=Mock(return_value=fault == 'monitor'))
    control.last_sync = monotonic() - (20 if fault == 'stalled' else 0)
    control.error = 'failed' if fault == 'error' else None
    durable = replace(snapshot, generation='new') if fault == 'generation' else snapshot
    control.read_state = AsyncMock(return_value={'phase': 'active', 'active': asdict(durable)})
    if fault == 'database':
        control.read_state.side_effect = ConnectionError('offline')
    try:
        assert await control.is_ready() is (fault == 'none')
        if fault == 'none':
            async with registry.corpus_operation():
                control.error = 'synchronization failed after admission'
                async with registry.corpus_operation():
                    assert registry.qdrant_client().snapshot == snapshot
        else:
            with pytest.raises(ValidationError, match='Corpus'):
                async with registry.corpus_operation():
                    pytest.fail('stale operation admitted')
            if fault == 'generation':
                registry.install_corpus(BoundQdrantClient(Mock(), durable), object())
                assert await control.is_ready()
                async with registry.corpus_operation():
                    assert registry.qdrant_client().snapshot == durable
    finally:
        await registry.close()


@pytest.mark.parametrize(
    "text,vector,valid",
    [
        ("private text", None, False),
        ("я !", None, False),
        ("private text", models.SparseVector(indices=[], values=[]), False),
        ("я !", models.SparseVector(indices=[], values=[]), True),
    ],
)
def test_corpus_requires_sparse_vector(text: str, vector: models.SparseVector | None, valid: bool) -> None:
    client, identity, row = fixture_corpus()
    try:
        row["content"] = text
        point = client.retrieve("docs_v1", [1], with_payload=True)[0]
        payload = point.payload
        payload["page_content"] = text
        vectors = {"": [1.0, 0.0]}
        if vector is not None:
            vectors[SPARSE_VECTOR_NAME] = vector
        client.upsert("docs_v1", [models.PointStruct(id=1, vector=vectors, payload=payload)])
        if valid:
            validate_corpus(client, "http://qdrant:6333", "docs_v1", identity, 2, [row])
        else:
            with pytest.raises(ValidationError, match="sparse vector"):
                validate_corpus(client, "http://qdrant:6333", "docs_v1", identity, 2, [row])
    finally:
        client.close()


@pytest.mark.parametrize("modifier", ["missing", None, models.Modifier.IDF])
def test_bootstrap_checks_sparse_config_for_empty_collection(
    monkeypatch: pytest.MonkeyPatch, modifier,
) -> None:
    monkeypatch.setattr(settings, "embed_dim", 2)
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            "docs",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
            sparse_vectors_config=(
                None if modifier == "missing"
                else {SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=modifier)}
            ),
        )
        snapshot = CorpusSnapshot("http://qdrant:6333", "docs", "docs", "identity")
        if modifier != models.Modifier.IDF:
            with pytest.raises(ValidationError, match="sparse index"):
                probe_active_corpus(client, snapshot, completed=False)
        else:
            probe_active_corpus(client, snapshot, completed=False)
    finally:
        client.close()
