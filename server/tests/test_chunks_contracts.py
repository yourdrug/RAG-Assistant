"""Public feature exports match their implementations."""

import importlib

import pytest


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.chunks", "application.chunks.mutations", "ChunkMutationService"),
        ("application.chunks", "application.chunks.queries", "ChunkQueryService"),
        ("application.chunks", "application.chunks.ports.settings", "ChunkSettingsPort"),
        ("application.chunks", "application.chunks.dto", "ChunkItemDTO"),
        ("application.chunks", "application.chunks.dto", "EditChunkResult"),
        ("application.chunks", "application.chunks.dto", "AddChunkResult"),
    ],
)
def test_public_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


@pytest.mark.parametrize(
    "name", ["load_doc_for_edit", "load_doc_for_add", "to_chunk_dto", "validate_chunk_acl"]
)
def test_access_and_mapping_helpers_are_internal(name: str) -> None:
    public = importlib.import_module("application.chunks")
    assert name not in public.__all__
    assert not hasattr(public, name)
