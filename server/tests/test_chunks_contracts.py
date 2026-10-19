"""Public feature exports match their implementations."""

import importlib

import pytest


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.chunks", "application.chunks.access", "load_doc_for_edit"),
        ("application.chunks", "application.chunks.access", "load_doc_for_add"),
        ("application.chunks", "application.chunks.mappers", "to_chunk_dto"),
        ("application.chunks", "application.chunks.mutations", "ChunkMutationService"),
        ("application.chunks", "application.chunks.queries", "validate_chunk_acl"),
        ("application.chunks", "application.chunks.queries", "ChunkQueryService"),
        ("application.chunks", "application.chunks.service", "ChunkService"),
        ("application.chunks", "application.chunks.settings", "ChunkSettingsPort"),
        ("application.chunks", "application.chunks.dto", "ChunkItemDTO"),
        ("application.chunks", "application.chunks.dto", "EditChunkResult"),
        ("application.chunks", "application.chunks.dto", "AddChunkResult"),
    ],
)
def test_public_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )
