from unittest.mock import AsyncMock, Mock
from dataclasses import asdict
from time import monotonic
from infrastructure.corpus.control import PostgresCorpusControl
import pytest
from infrastructure.corpus.snapshot import CorpusSnapshot, BoundQdrantClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
from dataclasses import replace
from domain.exceptions import ValidationError
from infrastructure.corpus.validation import validate_corpus
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.answer_cache import _cache_key
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.persistence import BM25LoadError


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
    client.create_collection('docs_v1', vectors_config=VectorParams(size=2, distance=Distance.COSINE))
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
                vector=[1.0, 0.0],
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
async def test_invalidated_bm25_is_reloaded_before_request_pins_corpus():
    index = BM25Index(['restored sparse document'])

    async def load():
        return index

    registry = MLClientRegistry(bm25_loader=load)
    snapshot = CorpusSnapshot('http://qdrant:6333', 'docs', 'v1', 'i', 'g1')
    registry.install_corpus(BoundQdrantClient(Mock(), snapshot), BM25Index(['old']))
    try:
        async with registry.corpus_operation():
            registry.invalidate_bm25()
            assert (await registry.ensure_bm25_loaded()).texts == ['old']
        async with registry.corpus_operation():
            assert await registry.ensure_bm25_loaded() is index
            assert registry.qdrant_client().snapshot == snapshot
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_corpus_switch_during_bm25_reload_pins_new_index():
    registry = MLClientRegistry()
    old = CorpusSnapshot('http://qdrant:6333', 'docs', 'v1', 'i', 'g1')
    new = replace(old, physical_collection='v2', generation='g2')
    current_index = BM25Index(['new corpus'])

    async def load():
        registry.install_corpus(BoundQdrantClient(Mock(), new), current_index)
        return BM25Index(['stale download'])

    registry._bm25_loader = load
    registry.install_corpus(BoundQdrantClient(Mock(), old), BM25Index(['old corpus']))
    registry.invalidate_bm25()
    try:
        async with registry.corpus_operation():
            assert registry.qdrant_client().snapshot == new
            assert await registry.ensure_bm25_loaded() is current_index
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_failed_bm25_reload_degrades_then_retries_next_request():
    calls = 0
    index = BM25Index(['recovered'])

    async def load():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BM25LoadError('storage unavailable')
        return index

    registry = MLClientRegistry(bm25_loader=load)
    snapshot = CorpusSnapshot('http://qdrant:6333', 'docs', 'v1', 'i', 'g1')
    registry.install_corpus(BoundQdrantClient(Mock(), snapshot), None)
    registry.invalidate_bm25()
    try:
        async with registry.corpus_operation():
            assert await registry.ensure_bm25_loaded() is None
        async with registry.corpus_operation():
            assert await registry.ensure_bm25_loaded() is index
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
    from infrastructure.corpus.control import validate_corpus_endpoint

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
    control = module.PostgresCorpusControl(None, registry, listener)
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
    index = object()
    monkeypatch.setattr(module, 'raw_corpus_client', Mock(return_value=Mock()))
    monkeypatch.setattr(module, 'build_corpus_bm25', Mock(return_value=index))
    monkeypatch.setattr(module.corpus_snapshot, 'installed_corpus', None)
    monkeypatch.setattr(settings, 'qdrant_url', old.endpoint)
    monkeypatch.setattr(settings, 'collection_name', old.alias)
    control.prepare = AsyncMock(return_value=(BoundQdrantClient(Mock(), candidate), object()))
    try:
        await control.tick()
        assert registry.corpus_snapshot == active
        assert await registry.ensure_bm25_loaded() is index
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
    control = PostgresCorpusControl(None, registry, None)
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
