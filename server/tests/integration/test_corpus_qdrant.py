"""A real Qdrant alias change keeps old readers on their physical corpus and ACL."""

import hashlib
import os
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    PointStruct,
    VectorParams,
)

from domain.exceptions import ValidationError
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.corpus.snapshot import BoundQdrantClient
from infrastructure.corpus.validation import validate_corpus
from infrastructure.repositories.vector.acl import build_qdrant_filter


@pytest.mark.integration
def test_alias_switch_preserves_physical_snapshot_and_private_acl():
    endpoint = os.environ.get("TEST_QDRANT_URL")
    if not endpoint:
        pytest.skip("Set TEST_QDRANT_URL to an isolated test Qdrant")
    client = QdrantClient(url=endpoint, timeout=5)
    prefix = "configuration_test_" + uuid4().hex
    alias = prefix + "_active"
    names = [prefix + "_v1", prefix + "_v2"]
    identity = "test:model:revision"
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    rows = [
        {
            "id": i,
            "document_id": i,
            "visibility": DocumentVisibility.INTERNAL_PRIVATE.value,
            "owner_id": owner,
            "group_id": None,
            "content_hash": f"hash-{i}",
            "doc_domain": "general",
            "content": f"private document {i}",
        }
        for i, owner in [(1, 7), (2, 8)]
    ]
    created = []
    try:
        for name in names:
            client.create_collection(name, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
            created.append(name)
            binding = name + "__embedding_identity"
            client.create_collection(
                binding, vectors_config={fingerprint: VectorParams(size=1, distance=Distance.COSINE)}
            )
            created.append(binding)
            client.upsert(
                name,
                [
                    PointStruct(
                        id=row["id"],
                        vector=[1.0, 0.0],
                        payload={
                            "page_content": row["content"],
                            "metadata": {
                                **{key: value for key, value in row.items() if key not in {"id", "content"}},
                                "embedding_model": identity,
                            },
                        },
                    )
                    for row in rows
                ],
                wait=True,
            )
        client.update_collection_aliases(
            [CreateAliasOperation(create_alias=CreateAlias(collection_name=names[0], alias_name=alias))]
        )
        old = validate_corpus(client, endpoint, alias, identity, 2, rows)
        bound = BoundQdrantClient(client, old)
        client.update_collection_aliases(
            [
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)),
                CreateAliasOperation(create_alias=CreateAlias(collection_name=names[1], alias_name=alias)),
            ]
        )
        new = validate_corpus(client, endpoint, alias, identity, 2, rows)
        assert old.physical_collection == names[0]
        assert new.physical_collection == names[1]
        assert old.namespace != new.namespace
        assert bound.snapshot is old
        with pytest.raises(ValidationError, match="metadata differs"):
            validate_corpus(client, endpoint, alias, identity, 2, [{**rows[0], "owner_id": 9}, rows[1]])
        client.update_collection_aliases([DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias))])
        # A pinned reader uses the physical corpus even when the alias no longer exists.
        for user_id in (7, 8, 9):
            access = build_qdrant_filter(
                {"id": user_id, "kind": UserKind.INTERNAL, "role": UserRole.USER}, []
            )
            points = bound.search(collection_name=alias, query_vector=[1.0, 0.0], query_filter=access)
            assert [point.id for point in points] == ({7: [1], 8: [2]}.get(user_id, []))
    finally:
        try:
            if any(item.alias_name == alias for item in client.get_aliases().aliases):
                client.update_collection_aliases(
                    [DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias))]
                )
            for name in reversed(created):
                client.delete_collection(name)
        finally:
            client.close()
