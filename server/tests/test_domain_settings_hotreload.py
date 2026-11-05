"""Tests for domain settings hot-reload (DomainSettingsAdapter + ConfigService)."""

from __future__ import annotations

from pathlib import Path
import sys

from application.runtime.config import ConfigParameter
from domain.shared.exceptions import EntityNotFoundError


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from typing import TYPE_CHECKING

import pytest

from application.admin import ConfigService
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from application.runtime.config import ConfigParameterChanged


class TestDomainSettingsAdapter:
    def test_roundtrip(self) -> None:
        adapter = DomainSettingsAdapter()
        adapter.replace_snapshot({("max_unit_chars", "decree"): "1200"})
        assert adapter.get("max_unit_chars", domain_key="decree") == "1200"

    def test_missing_key_raises_loudly(self) -> None:
        adapter = DomainSettingsAdapter()
        adapter.replace_snapshot({("some_global", "legal"): "1"})
        with pytest.raises(KeyError):
            adapter.get("max_unit_chars", domain_key="decree")

    def test_no_hidden_cross_domain_fallback(self) -> None:
        """A value in another domain must never be silently reused."""
        adapter = DomainSettingsAdapter()
        adapter.replace_snapshot({("max_unit_chars", "legal"): "999"})
        with pytest.raises(KeyError):
            adapter.get("max_unit_chars", domain_key="decree")

    def test_snapshot_and_get(self) -> None:
        adapter = DomainSettingsAdapter()
        adapter.replace_snapshot({("max_unit_chars", "legal"): "1500"})
        assert adapter.get("max_unit_chars", domain_key="legal") == "1500"

    def test_replace_snapshot_removes_old_values_and_copies_input(self) -> None:
        adapter = DomainSettingsAdapter()
        adapter.replace_snapshot({("old", "decree"): "1"})
        values = {("new", "legal"): "2"}
        adapter.replace_snapshot(values)
        values[("new", "legal")] = "changed"
        exported = adapter.snapshot()
        exported[("new", "legal")] = "also changed"
        assert adapter.get("new", "legal") == "2"
        with pytest.raises(KeyError):
            adapter.get("old", "decree")


class TestRegistryWithAdapter:
    def test_threshold_from_settings_drives_general_fallback(self) -> None:
        """Unknown documents fall to general when the DB threshold is not met."""
        adapter = DomainSettingsAdapter()
        profile = DecreeDomainProfile(settings=adapter)
        registry = DomainProfileRegistry()
        registry.register(profile)

        # Simulate startup load from DB
        adapter.replace_snapshot(
            {(default.key, "decree"): default.value for default in profile.config_defaults()}
        )

        text = "УКАЗ\nПОСТАНОВЛЯЮ:\n"  # weak signal: 3.0+2.0 but no points
        result = registry.classify(text, settings=adapter)
        # Score below seeded threshold (2.0 is met actually) — use even weaker text
        weak = "Просто деловое письмо без структуры указа. " * 5
        result = registry.classify(weak, settings=adapter)
        assert result.domain_key == "general"


class TestConfigServiceDomainParams:
    @pytest.mark.asyncio
    async def test_update_domain_parameter_publishes_with_domain_key(self) -> None:
        factory = FakeUnitOfWorkFactory()
        uow = factory._uow
        await uow.config_parameters.save(_param("max_unit_chars", "1200", domain_key="decree"))

        events: list[ConfigParameterChanged] = []
        bus = type("Bus", (), {"publish": staticmethod(events.append)})()
        service = ConfigService(uow_factory=factory, event_bus=bus)

        await service.update_parameter("max_unit_chars", "2000", changed_by=1, domain_key="decree")

        assert len(events) == 1
        assert events[0].domain_key == "decree"
        assert events[0].new_value == "2000"
        # DB row updated
        saved = await uow.config_parameters.get_by_key_and_domain("max_unit_chars", "decree")
        assert saved is not None
        assert saved.value == "2000"

    @pytest.mark.asyncio
    async def test_update_unknown_domain_param_raises(self) -> None:
        service = ConfigService(
            uow_factory=FakeUnitOfWorkFactory(),
            event_bus=type("Bus", (), {"publish": staticmethod(lambda _e: None)})(),
        )

        with pytest.raises(EntityNotFoundError):
            await service.update_parameter("max_unit_chars", "1", domain_key="decree")


def _param(key: str, value: str, domain_key: str | None) -> ConfigParameter:

    return ConfigParameter(
        key=key,
        value=value,
        value_type="int",
        category=f"domain:{domain_key}" if domain_key else "rag",
        description="test",
        domain_key=domain_key,
    )
