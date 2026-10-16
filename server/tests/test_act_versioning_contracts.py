"""Compatibility and forwarding across the act versioning package boundary."""

from __future__ import annotations

from datetime import date
import importlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from application.act_versioning import ActVersioningService


if TYPE_CHECKING:
    from application.act_versioning import VersionUpdateOptions


LEGACY_EXPORTS = [
    ("application.dto.versioning_dto", ("VersioningPlan", "VersioningResult")),
    ("application.services.act_version_answers", ("ActVersionAnswerInvalidator",)),
    ("application.services.act_version_commands", ("ActVersionCommands",)),
    ("application.services.act_version_persistence", ("ActVersionPersistence",)),
    ("application.services.act_version_planner", ("ActVersionPlanner",)),
    ("application.services.act_version_queries", ("ActVersionQueries", "ActSummary")),
    ("application.services.act_versioning_service", ("ActVersioningService",)),
]


@pytest.mark.parametrize("legacy,names", LEGACY_EXPORTS)
def test_legacy_exports_share_canonical_objects(legacy: str, names: tuple[str, ...]) -> None:
    old = importlib.import_module(legacy)
    canonical = importlib.import_module("application.act_versioning")

    for name in names:
        assert getattr(old, name) is getattr(canonical, name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"effective_from": None, "effective_from_provided": True},
        {"effective_to": None, "effective_to_provided": True},
        {"act_id": None, "act_id_provided": True, "verified_by": 7},
        {"effective_from": date(2024, 1, 1), "effective_to": date(2025, 1, 1), "act_id": 8},
    ],
)
async def test_update_preserves_omitted_and_explicit_fields(options: VersionUpdateOptions) -> None:
    service = ActVersioningService(Mock(), Mock())
    commands = AsyncMock()
    service._commands = commands

    await service.update_version(42, **options)

    commands.update_version.assert_awaited_once_with(version_id=42, **options)


@pytest.mark.asyncio
async def test_creation_forwards_callers_transaction_and_date_options() -> None:
    service = ActVersioningService(Mock(), Mock())
    commands = AsyncMock()
    service._commands = commands
    uow = Mock()
    profile = Mock()
    effective_date = date(2024, 1, 1)

    result = await service.create_version_in_uow(
        uow, profile, 42, [], effective_date=effective_date, date_confidence=0.9
    )

    assert result is commands.create_version_in_uow.return_value
    commands.create_version_in_uow.assert_awaited_once_with(
        uow=uow,
        profile=profile,
        document_id=42,
        extracted_refs=[],
        effective_date=effective_date,
        date_confidence=0.9,
    )
