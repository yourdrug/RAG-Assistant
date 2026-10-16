"""Compatibility of legacy exports with the canonical feature package."""

import importlib

import pytest


@pytest.mark.parametrize(
    "legacy,canonical,name",
    [
        ("application.services.chunk_access", "application.chunks.access", "load_doc_for_edit"),
        ("application.services.chunk_access", "application.chunks.access", "load_doc_for_add"),
        ("application.services.chunk_mappers", "application.chunks.mappers", "to_chunk_dto"),
        (
            "application.services.chunk_mutation_service",
            "application.chunks.mutations",
            "ChunkMutationService",
        ),
        ("application.services.chunk_query_service", "application.chunks.queries", "validate_chunk_acl"),
        ("application.services.chunk_query_service", "application.chunks.queries", "ChunkQueryService"),
        ("application.services.chunk_service", "application.chunks.service", "ChunkService"),
        ("application.ports.chunk_settings", "application.chunks.settings", "ChunkSettingsPort"),
        ("application.dto.chunk_dto", "application.chunks.dto", "ChunkItemDTO"),
        ("application.dto.chunk_dto", "application.chunks.dto", "EditChunkResult"),
        ("application.dto.chunk_dto", "application.chunks.dto", "AddChunkResult"),
    ],
)
def test_legacy_exports_share_canonical_objects(legacy: str, canonical: str, name: str) -> None:
    assert getattr(importlib.import_module(legacy), name) is getattr(importlib.import_module(canonical), name)
