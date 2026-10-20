"""Retryable PostgreSQL journal, write barrier, and serving-process acknowledgements.

A session advisory lock fences Qdrant writes without a database transaction
around network probes. A short final transaction fences PostgreSQL mutations
and rechecks the corpus. Committed switches remain write-blocked until API and
worker have installed the same immutable resource generation.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import logging
import re
from uuid import uuid4
from time import monotonic
from urllib.parse import urlsplit

import psycopg
from sqlalchemy import text
from qdrant_client import QdrantClient
from qdrant_client.models import Modifier

from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME

from config import settings
from domain.shared.exceptions import ValidationError
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import OutboxStatus
from infrastructure.ml.config.embedding_identity import configured_embedding_identity
from infrastructure.ml.config.endpoint_policy import endpoint_origin
from infrastructure.corpus import snapshot as corpus_snapshot
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.corpus.validation import (
    corpus_digest,
    validate_corpus,
    validate_corpus_binding,
)

log = logging.getLogger('default')
CORPUS_LOCK = 7319048273461
CHUNKS_SQL = '''SELECT id, document_id, visibility, owner_id, group_id, content_hash,
                      doc_domain, content FROM chunks ORDER BY id'''


def connection_arguments():
    return {
        "host": settings.db_host,
        "port": int(settings.db_port),
        "user": settings.db_user,
        "password": settings.db_password,
        "dbname": settings.db_name,
        "connect_timeout": 5,
    }


@contextmanager
def corpus_write_guard(snapshot):
    # Called by the synchronous Qdrant adapter in its existing worker thread.
    with psycopg.connect(**connection_arguments(), autocommit=True) as conn:
        conn.execute("SET statement_timeout = '5s'")
        conn.execute('SELECT pg_advisory_lock_shared(%s)', (CORPUS_LOCK,))
        admitted = False
        try:
            row = conn.execute('SELECT phase, active FROM corpus_control WHERE id=1').fetchone()
            if row is None or row[0] not in {'active', 'failed'}:
                raise ValidationError('Corpus switch is pending; retry the write after readiness')
            if CorpusSnapshot(**row[1]).namespace != snapshot.namespace:
                raise ValidationError('Corpus generation changed; retry using the current snapshot')
            admitted = True
            yield
        finally:
            try:
                if admitted:
                    conn.execute('UPDATE corpus_control SET write_epoch=write_epoch+1 WHERE id=1')
            finally:
                conn.execute('SELECT pg_advisory_unlock_shared(%s)', (CORPUS_LOCK,))


def raw_corpus_client(endpoint):
    allowed = {endpoint_origin(value) for value in settings.ml_internal_allowed_origins}
    if endpoint_origin(endpoint) not in allowed or urlsplit(endpoint).path not in {'', '/'}:
        raise ValidationError('Qdrant origin must be deployment-approved before credentials are used')
    return QdrantClient(
        url=endpoint, api_key=settings.qdrant_api_key, timeout=settings.qdrant_timeout, follow_redirects=False
    )


def validate_corpus_endpoint(endpoint, current):
    allowed = {endpoint_origin(current)} | {endpoint_origin(v) for v in settings.ml_internal_allowed_origins}
    if endpoint_origin(endpoint) not in allowed or urlsplit(endpoint).path not in {'', '/'}:
        raise ValidationError('Qdrant origin is not deployment-approved')


class PostgresCorpusControl:
    def __init__(self, session_factory, registry, configuration_listener):
        self.session_factory = session_factory
        self.registry = registry
        self.listener = configuration_listener
        self.process_id = str(uuid4())
        self.role = 'cli'
        self.task = None
        self.prepared = None
        self.prepared_index = None
        self.error = None
        self.sync_lock = asyncio.Lock()
        self.last_sync: float | None = None
        self.registry.corpus_readiness = self.is_ready

    async def read_state(self):
        async with self.session_factory() as session:
            return (
                (await session.execute(text('SELECT * FROM corpus_control WHERE id=1')))
                .mappings()
                .one_or_none()
            )

    async def chunks(self):
        async with self.session_factory() as session:
            return list((await session.execute(text(CHUNKS_SQL))).mappings())

    async def initial_snapshot(self):
        client = raw_corpus_client(settings.qdrant_url)
        try:
            aliases = await asyncio.to_thread(client.get_aliases)
            physical = next(
                (a.collection_name for a in aliases.aliases if a.alias_name == settings.collection_name),
                settings.collection_name,
            )
            rows = await self.chunks()
            names = {item.name for item in (await asyncio.to_thread(client.get_collections)).collections}
            if physical in names and (await asyncio.to_thread(client.get_collection, physical)).points_count:
                verified = await asyncio.to_thread(
                    validate_corpus,
                    client,
                    settings.qdrant_url,
                    settings.collection_name,
                    configured_embedding_identity(),
                    settings.embed_dim,
                    rows,
                )
                return replace(verified, generation=str(uuid4()))
            return CorpusSnapshot(
                settings.qdrant_url,
                settings.collection_name,
                physical,
                configured_embedding_identity(),
                str(uuid4()),
            )
        finally:
            await asyncio.to_thread(client.close)

    async def start(self, role):
        self.role = role
        state = await self.read_state()
        if state is None:
            snapshot = await self.initial_snapshot()
            async with self.session_factory() as session, session.begin():
                await session.execute(
                    text('''INSERT INTO corpus_control (id, phase, active)
                    VALUES (1, 'active', CAST(:active AS jsonb)) ON CONFLICT DO NOTHING'''),
                    {'active': json.dumps(asdict(snapshot))},
                )
            state = await self.read_state()
        active = CorpusSnapshot(**state['active'])
        if (settings.qdrant_url, settings.collection_name) != (active.endpoint, active.alias):
            raise ValidationError('Saved Qdrant settings differ from the durable corpus journal')
        if active.embedding_identity != configured_embedding_identity():
            raise ValidationError('Active corpus embedding identity differs from the configured encoder')
        raw = raw_corpus_client(active.endpoint)
        try:
            async with self.session_factory() as session:
                completed = (
                    await session.execute(
                        text('''SELECT EXISTS(SELECT 1 FROM chunks c
                        JOIN documents d ON d.id=c.document_id WHERE d.status=:status)'''),
                        {'status': DocumentStatus.DONE.value},
                    )
                ).scalar_one()
            await asyncio.to_thread(probe_active_corpus, raw, active, completed)
            index = None
        except BaseException:
            await asyncio.to_thread(raw.close)
            raise
        self.install(BoundQdrantClient(raw, active, corpus_write_guard), index)
        await self.tick()
        self.task = asyncio.create_task(self.monitor(), name=f'corpus-control-{role}')

    def install(self, client, index):
        self.registry.install_corpus(client, index)
        corpus_snapshot.installed_corpus = client.snapshot
        # This synchronous publication is the sole owner of corpus settings.
        settings.qdrant_url = client.snapshot.endpoint
        settings.collection_name = client.snapshot.alias

    async def heartbeat(self, prepared=None, error=None):
        active = self.registry.corpus_snapshot
        async with self.session_factory() as session, session.begin():
            await session.execute(
                text('''
                INSERT INTO corpus_processes (process_id, role, prepared_generation,
                                              applied_generation, error, heartbeat_at)
                VALUES (:id, :role, :prepared, :applied, :error, now())
                ON CONFLICT (process_id) DO UPDATE SET
                    prepared_generation=EXCLUDED.prepared_generation,
                    applied_generation=EXCLUDED.applied_generation, error=EXCLUDED.error,
                    heartbeat_at=now()
            '''),
                {
                    'id': self.process_id,
                    'role': self.role,
                    'prepared': prepared,
                    'applied': active.generation if active else None,
                    'error': error,
                },
            )

    async def prepare(self, candidate, expected_digest):
        rows = await self.chunks()
        if corpus_digest(rows) != expected_digest:
            raise ValidationError('PostgreSQL corpus changed during preparation')
        validate_corpus_endpoint(candidate.endpoint, settings.qdrant_url)
        raw = raw_corpus_client(candidate.endpoint)
        try:
            verified = await asyncio.to_thread(
                validate_corpus_binding,
                raw,
                candidate.endpoint,
                candidate.alias,
                candidate.embedding_identity,
                settings.embed_dim,
            )
            if verified.physical_collection != candidate.physical_collection:
                raise ValidationError('Qdrant alias changed during preparation')
            index = None
            return BoundQdrantClient(raw, candidate, corpus_write_guard), index
        except BaseException:
            await asyncio.to_thread(raw.close)
            raise

    async def tick(self):
        async with self.sync_lock:
            state = await self.read_state()
            if state is None:
                raise ValidationError('Corpus journal is not initialized')
            active = CorpusSnapshot(**state['active'])
            if self.registry.corpus_snapshot.generation != active.generation or self.error:
                # A replica may miss the entire switch, including its committing phase.
                await self.apply_candidate({**state, 'phase': 'committing'}, active)
                if self.error:
                    await self.heartbeat(error=self.error)
                    return
            candidate = CorpusSnapshot(**state['candidate']) if state['candidate'] else None
            prepared_id = None
            if state['phase'] in {'preparing', 'committing'} and candidate is not None:
                prepared_id = await self.apply_candidate(state, candidate)
            else:
                self.error = None
                if self.prepared is not None:
                    await asyncio.to_thread(self.prepared.close)
                    self.prepared = self.prepared_index = None
            await self.heartbeat(prepared_id, self.error)
            await self.recover(state)
            self.last_sync = monotonic()

    async def apply_candidate(self, state, candidate):
        prepared_id = None
        try:
            current = self.registry.corpus_snapshot
            if current.generation != candidate.generation:
                if self.prepared is None or self.prepared.snapshot != candidate:
                    if self.prepared is not None:
                        await asyncio.to_thread(self.prepared.close)
                    self.prepared = self.prepared_index = None
                    if state['phase'] == 'committing':
                        # The durable active journal is the admission proof; the old
                        # preparation digest can be stale after subsequent ingestion.
                        index = None
                        self.prepared = BoundQdrantClient(
                            raw_corpus_client(candidate.endpoint), candidate, corpus_write_guard
                        )
                        self.prepared_index = index
                    else:
                        self.prepared, self.prepared_index = await self.prepare(candidate, state['digest'])
                prepared_id = candidate.generation
                if state['phase'] == 'committing':
                    self.install(self.prepared, self.prepared_index)
                    self.prepared = self.prepared_index = None
                    if self.listener is not None:
                        await self.listener.synchronize()
            else:
                prepared_id = candidate.generation
                if self.error and self.listener is not None:
                    await self.listener.synchronize()
            self.error = None
        except Exception:
            self.error = 'Corpus preparation/application failed'
            log.exception('Corpus preparation/application failed')
        return prepared_id

    async def participants(self):
        async with self.session_factory() as session:
            return list(
                (
                    await session.execute(
                        text('''SELECT *,
                heartbeat_at > now() - interval '15 seconds' AS alive
                FROM corpus_processes WHERE role IN ('api', 'worker')''')
                    )
                ).mappings()
            )

    async def recover(self, state):
        participants = await self.participants()
        live = [p for p in participants if p['alive']]
        if state['phase'] == 'committing' and state['candidate']:
            generation = state['candidate']['generation']
            if {p['role'] for p in live} == {'api', 'worker'} and all(
                p['applied_generation'] == generation and not p['error'] for p in live
            ):
                async with self.session_factory() as session, session.begin():
                    await session.execute(
                        text('''UPDATE corpus_control SET phase='active', error=NULL,
                        candidate=NULL, updated_at=now() WHERE id=1 AND phase='committing'
                        AND active->>'generation'=:generation'''),
                        {'generation': generation},
                    )
        elif state['phase'] == 'preparing' and not any(
            p['process_id'] == state['owner'] and p['alive'] for p in live
        ):
            async with self.session_factory() as session, session.begin():
                await session.execute(
                    text('''UPDATE corpus_control SET phase='failed',
                    error='Switch owner unavailable before commit', candidate=NULL, updated_at=now()
                    WHERE id=1 AND phase='preparing' AND owner=:owner'''),
                    {'owner': state['owner']},
                )

    async def monitor(self):
        while True:
            await asyncio.sleep(2)
            try:
                await self.tick()
            except Exception:
                self.error = 'Corpus synchronization unavailable'
                log.exception('Corpus synchronization unavailable')

    async def is_ready(self) -> bool:
        if (
            self.task is None
            or self.task.done()
            or self.error
            or self.last_sync is None
            or monotonic() - self.last_sync >= 15
        ):
            return False
        try:
            state = await self.read_state()
        except Exception:
            log.exception('Corpus admission journal unavailable')
            return False
        current = self.registry.corpus_snapshot
        return bool(state and current is not None and current.generation == state['active']['generation'])

    async def status(self):
        state = await self.read_state()
        return {
            'phase': state['phase'] if state else 'uninitialized',
            'active': state['active'] if state else None,
            'candidate': state['candidate'] if state else None,
            'error': state['error'] if state else None,
            'processes': [dict(p) for p in await self.participants()],
        }

    async def quiescent(self, session):
        busy = (
            await session.execute(
                text('''SELECT
            EXISTS(SELECT 1 FROM documents WHERE status IN (:pending,:processing,:indexing))
            OR EXISTS(SELECT 1 FROM vector_store_outbox WHERE status <> :done)'''),
                {
                    'pending': DocumentStatus.PENDING.value,
                    'processing': DocumentStatus.PROCESSING.value,
                    'indexing': DocumentStatus.INDEXING.value,
                    'done': OutboxStatus.DONE.value,
                },
            )
        ).scalar_one()
        if busy:
            raise ValidationError('Finish or reconcile ingestion/outbox work before switching corpus')

    async def switch(self, endpoint, collection, expected_generation):
        validate_corpus_endpoint(endpoint, settings.qdrant_url)
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,255}', collection):
            raise ValidationError('Invalid Qdrant collection or alias')
        conn = None
        committed = False
        try:
            state = await self.read_state()
            if state is None or state['phase'] not in {'active', 'failed'}:
                raise ValidationError('A corpus switch is already pending; inspect its status')
            if state['active']['generation'] != expected_generation:
                raise ValidationError('Corpus generation changed; refresh before switching')
            participants = [p for p in await self.participants() if p['alive']]
            if {p['role'] for p in participants} != {'api', 'worker'} or any(
                p['error'] or p['applied_generation'] != expected_generation for p in participants
            ):
                raise ValidationError('Both API and worker must be ready before switching')
            async with self.session_factory() as session:
                await self.quiescent(session)
                rows = list((await session.execute(text(CHUNKS_SQL))).mappings())
                config_rows = (
                    await session.execute(text('SELECT key, domain_key, value FROM config_parameters'))
                ).all()
            raw = raw_corpus_client(endpoint)
            try:
                snapshot = await asyncio.to_thread(
                    validate_corpus,
                    raw,
                    endpoint,
                    collection,
                    configured_embedding_identity(),
                    settings.embed_dim,
                    rows,
                )
                snapshot = replace(snapshot, generation=str(uuid4()))
            finally:
                await asyncio.to_thread(raw.close)
            conn = await psycopg.AsyncConnection.connect(**connection_arguments(), autocommit=True)
            await conn.execute("SET statement_timeout = '5s'")
            await conn.execute('SELECT pg_advisory_lock(%s)', (CORPUS_LOCK,))
            latest_state = await self.read_state()
            validate_switch_version(latest_state, state, expected_generation)
            digest = corpus_digest(rows)
            async with self.session_factory() as session, session.begin():
                await session.execute(
                    text('''UPDATE corpus_control SET phase='preparing',
                    candidate=CAST(:candidate AS jsonb), digest=:digest, owner=:owner,
                    error=NULL, updated_at=now() WHERE id=1'''),
                    {'candidate': json.dumps(asdict(snapshot)), 'digest': digest, 'owner': self.process_id},
                )
            async with asyncio.timeout(15):
                await self.wait_prepared(snapshot.generation, {p['process_id'] for p in participants})
            await self.commit_switch(snapshot, digest, config_rows)
            committed = True
            for _ in range(30):
                await self.tick()
                state = await self.read_state()
                if state['phase'] == 'active':
                    return await self.status()
                await asyncio.sleep(0.5)
            raise ValidationError(
                'Corpus committed; writes remain fenced until API/worker readiness recovers'
            )
        except BaseException:
            if not committed:
                async with self.session_factory() as session, session.begin():
                    await session.execute(
                        text('''UPDATE corpus_control SET phase='failed', candidate=NULL,
                        error='Corpus switch failed before commit', updated_at=now()
                        WHERE id=1 AND phase='preparing' AND owner=:owner'''),
                        {'owner': self.process_id},
                    )
            raise
        finally:
            if conn is not None:
                await conn.close()

    async def commit_switch(self, snapshot, digest, config_rows):
        # No network calls in this short final transaction.
        async with self.session_factory() as session, session.begin():
            journal = (
                (
                    await session.execute(
                        text('SELECT phase, owner, candidate FROM corpus_control WHERE id=1 FOR UPDATE')
                    )
                )
                .mappings()
                .one()
            )
            if (
                journal['phase'] != 'preparing'
                or journal['owner'] != self.process_id
                or not journal['candidate']
                or journal['candidate']['generation'] != snapshot.generation
            ):
                raise ValidationError('Corpus preparation journal changed before commit')
            await session.execute(
                text('''LOCK TABLE documents, chunks, vector_store_outbox,
                config_parameters IN SHARE ROW EXCLUSIVE MODE''')
            )
            await self.quiescent(session)
            latest = list((await session.execute(text(CHUNKS_SQL))).mappings())
            config_latest = (
                await session.execute(text('SELECT key, domain_key, value FROM config_parameters'))
            ).all()
            if corpus_digest(latest) != digest or set(config_latest) != set(config_rows):
                raise ValidationError('Configuration/corpus changed during probe; retry switching')
            for key, value in [('qdrant_url', snapshot.endpoint), ('collection_name', snapshot.alias)]:
                result = await session.execute(
                    text('''UPDATE config_parameters SET value=:value
                    WHERE key=:key AND domain_key IS NULL'''),
                    {'key': key, 'value': value},
                )
                if result.rowcount != 1:
                    raise ValidationError('Corpus configuration is not initialized')
            await session.execute(
                text('''UPDATE corpus_control SET phase='committing',
                active=candidate, updated_at=now() WHERE id=1''')
            )
            await session.execute(text("SELECT pg_notify('config_changed', '{}')"))

    async def wait_prepared(self, generation, required):
        for _ in range(30):
            await self.tick()
            participants = {p['process_id']: p for p in await self.participants()}
            if any(
                pid not in participants or not participants[pid]['alive'] or participants[pid]['error']
                for pid in required
            ):
                raise ValidationError('Serving process unavailable or failed corpus preparation')
            if all(participants[pid]['prepared_generation'] == generation for pid in required):
                return
            await asyncio.sleep(0.5)
        raise ValidationError('Serving processes did not prepare the corpus before timeout')

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        if self.prepared is not None:
            await asyncio.to_thread(self.prepared.close)
            self.prepared = None
        async with self.session_factory() as session, session.begin():
            await session.execute(
                text('DELETE FROM corpus_processes WHERE process_id=:id'), {'id': self.process_id}
            )


def validate_switch_version(latest, previous, expected_generation):
    if (
        latest['active']['generation'] != expected_generation
        or latest['phase'] not in {'active', 'failed'}
        or latest['write_epoch'] != previous['write_epoch']
    ):
        raise ValidationError('Corpus changed during validation; retry the switch')


def probe_active_corpus(client, snapshot, completed):
    names = {item.name for item in client.get_collections().collections}
    physical = snapshot.physical_collection
    if physical not in names:
        if completed:
            raise ValidationError('Completed PostgreSQL chunks have no active Qdrant collection')
        return  # Fresh empty deployment; ingestion owns collection creation.
    info = client.get_collection(physical)
    sparse = info.config.params.sparse_vectors or {}
    if SPARSE_VECTOR_NAME not in sparse or sparse[SPARSE_VECTOR_NAME].modifier != Modifier.IDF:
        raise ValidationError('Active corpus has no compatible BM25 sparse index')
    if not info.points_count and physical + '__embedding_identity' not in names:
        if completed or getattr(info.config.params.vectors, 'size', None) != settings.embed_dim:
            raise ValidationError('Active corpus has no compatible immutable identity')
        return
    validate_corpus_binding(
        client, snapshot.endpoint, physical, snapshot.embedding_identity, settings.embed_dim
    )
