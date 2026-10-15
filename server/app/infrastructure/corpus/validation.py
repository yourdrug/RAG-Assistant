"""Read-only validation: a candidate must match the PostgreSQL corpus and ACL."""

import hashlib

from domain.exceptions import ValidationError
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.corpus.snapshot import CorpusSnapshot


ACL_FIELDS = ('document_id', 'visibility', 'owner_id', 'group_id', 'content_hash', 'doc_domain')


def corpus_digest(rows) -> str:
    return hashlib.sha256(
        repr(sorted((row['id'], *(row[key] for key in ACL_FIELDS), row['content']) for row in rows)).encode()
    ).hexdigest()


def build_corpus_bm25(rows):
    return BM25Index(
        [row['content'] for row in rows],
        hashes=[row['content_hash'] for row in rows],
        doc_visibility=[row['visibility'] for row in rows],
        doc_owner_id=[row['owner_id'] for row in rows],
        doc_group_id=[row['group_id'] for row in rows],
    )


def validate_corpus(client, endpoint, name, identity, dimension, rows) -> CorpusSnapshot:
    snapshot = validate_corpus_binding(client, endpoint, name, identity, dimension)
    physical = snapshot.physical_collection
    expected = {row['id']: row for row in rows}
    seen: set[int | str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=physical, offset=offset, limit=256, with_payload=True, with_vectors=False
        )
        for point in points:
            validate_corpus_point(point, expected, seen, identity)
            seen.add(point.id)
        if offset is None:
            break
    if seen != set(expected):
        raise ValidationError('Selected corpus is missing PostgreSQL chunks')
    return CorpusSnapshot(endpoint, name, physical, identity)


def validate_corpus_point(point, expected, seen, identity):
    row = expected.get(point.id)
    payload = point.payload or {}
    metadata = payload.get('metadata') or {}
    if row is None or point.id in seen:
        raise ValidationError('Selected corpus contains unknown or duplicate chunk IDs')
    if any(key not in metadata or metadata[key] != row[key] for key in ACL_FIELDS):
        raise ValidationError('Selected corpus ACL/content metadata differs from PostgreSQL')
    if metadata.get('embedding_model') != identity or payload.get('page_content') != row['content']:
        raise ValidationError('Selected corpus content or embedding identity differs from PostgreSQL')


def validate_corpus_binding(client, endpoint, name, identity, dimension):
    names = {item.name for item in client.get_collections().collections}
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    physical = aliases.get(name, name)
    if physical not in names:
        raise ValidationError('Selected corpus does not exist')
    info = client.get_collection(physical)
    vectors = info.config.params.vectors
    if getattr(vectors, 'size', None) != dimension:
        raise ValidationError('Selected corpus has incompatible vector dimension')
    binding = physical + '__embedding_identity'
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    if binding not in names:
        raise ValidationError('Selected corpus has no immutable embedding identity')
    binding_vectors = client.get_collection(binding).config.params.vectors
    if not isinstance(binding_vectors, dict) or set(binding_vectors) != {fingerprint}:
        raise ValidationError('Selected corpus has incompatible embedding identity')
    return CorpusSnapshot(endpoint, name, physical, identity)
