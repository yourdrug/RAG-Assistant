"""Pytest fixtures of the Belarusian law pack, registered by conftest only when the pack imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from infrastructure.domain_packs.belarus_law.evidence.profile import BelarusEvidenceProfile
from infrastructure.ml.evidence_profiles.profiles import evidence_profile_scope
from tests.law_pack import law_registry as build_law_registry


if TYPE_CHECKING:
    from collections.abc import Iterator

    from domain.ingestion.profiles.registry import DomainProfileRegistry


@pytest.fixture(autouse=True)
def belarus_evidence_profile(request: pytest.FixtureRequest) -> Iterator[None]:
    """Select the law pack's evidence rules for tests marked ``belarus``."""
    if not request.node.get_closest_marker("belarus"):
        yield
        return

    with evidence_profile_scope(BelarusEvidenceProfile()):
        yield


@pytest.fixture
def law_registry() -> DomainProfileRegistry:
    """Document profiles of the Belarusian law pack; requesting it marks the test ``belarus``."""
    return build_law_registry()
