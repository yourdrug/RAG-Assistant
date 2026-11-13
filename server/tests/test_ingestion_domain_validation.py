"""Unsupported domain values must never be attached to ingestion chunks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from application.ingestion import S3DocumentLoader
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.enums import DocDomain
from infrastructure.domain_profile import register_pack_profiles
from infrastructure.domain_profile.packs import NEUTRAL_PACK
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter


class PassthroughSplitter:
    def split(self, docs: Any, context: object) -> Any:
        _ = context
        return docs


@pytest.fixture
def loader() -> S3DocumentLoader:
    # Neutral domain pack: only the core ``general`` profile is known.
    registry = DomainProfileRegistry()
    register_pack_profiles(registry, NEUTRAL_PACK, DomainSettingsAdapter())
    return S3DocumentLoader(
        MagicMock(), MagicMock(), PassthroughSplitter(), SimpleNamespace(), domain_registry=registry
    )


def test_unknown_explicit_domain_is_rejected_by_loader(loader: S3DocumentLoader) -> None:
    with pytest.raises(ValueError, match="legal_typo"):
        loader.classify_text_domain("document text", "legal_typo")


def test_direct_indexing_cannot_save_unknown_domain_metadata(loader: S3DocumentLoader) -> None:
    with pytest.raises(ValueError, match="legal_typo"):
        loader.index_docs([RawDocument("document text", {})], domain="legal_typo")


@pytest.mark.parametrize("domain", list(DocDomain))
def test_known_domain_is_preserved_in_chunks(loader: S3DocumentLoader, domain: DocDomain) -> None:
    resolved = loader.classify_text_domain("document text", domain.value)
    chunks = loader.index_docs([RawDocument("document text", {})], domain=resolved)
    assert chunks[0].metadata["doc_domain"] == domain.value


def test_law_pack_profile_is_rejected_by_neutral_pack(loader: S3DocumentLoader) -> None:
    with pytest.raises(ValueError, match="legal"):
        loader.classify_text_domain("document text", "legal")
